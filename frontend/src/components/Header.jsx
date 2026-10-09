import React, { useEffect, useState } from 'react';
import { Activity, ShieldCheck, Wifi, WifiOff } from 'lucide-react';

export default function Header() {
  const [health, setHealth] = useState({ online: false, device: 'Connecting...', params: 0 });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
          const data = await res.json();
          setHealth({
            online: true,
            device: data.device?.toUpperCase() || 'GPU',
            params: data.trainable_parameters || 4317825
          });
        } else {
          setHealth({ online: false, device: 'Offline', params: 0 });
        }
      } catch (err) {
        setHealth({ online: false, device: 'Offline', params: 0 });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="header-glass">
      <div className="brand-section">
        <div className="brand-icon-wrapper">
          <Activity size={24} />
        </div>
        <div>
          <h1 className="brand-title">SUSHRUTA</h1>
          <p className="brand-subtitle">
            AI-Assisted Explainable Medical Image Segmentation
          </p>
        </div>
      </div>

      <div className={`status-badge ${health.online ? 'online' : 'offline'}`}>
        <div className="pulse-dot"></div>
        {health.online ? (
          <>
            <Wifi size={14} />
            <span>Backend Connected ({health.device})</span>
          </>
        ) : (
          <>
            <WifiOff size={14} />
            <span>Backend Offline</span>
          </>
        )}
      </div>
    </header>
  );
}
