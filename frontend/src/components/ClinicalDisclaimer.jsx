import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function ClinicalDisclaimer({ text }) {
  return (
    <footer style={{
      marginTop: '3rem',
      padding: '1.25rem 1.5rem',
      background: 'rgba(15, 23, 42, 0.8)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-lg)',
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      fontSize: '0.8rem',
      color: 'var(--text-secondary)'
    }}>
      <ShieldAlert size={24} color="var(--accent-amber)" style={{ flexShrink: 0 }} />
      <div>
        <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.2rem' }}>
          ACADEMIC RESEARCH & DECISION-SUPPORT PROTOTYPE DISCLAIMER
        </strong>
        {text || 'This system is an academic/research prototype and is not intended to provide medical diagnosis or replace qualified clinical judgment.'}
        <span style={{ display: 'block', marginTop: '0.25rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
          * Baseline held-out test evaluation: Dice ≈ 99.59%. Performance on external hospital datasets is not clinically validated.
        </span>
      </div>
    </footer>
  );
}
