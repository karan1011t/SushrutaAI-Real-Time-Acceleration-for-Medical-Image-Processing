import React, { useState } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import VisualCard from './components/VisualCard';
import ImageModal from './components/ImageModal';
import ClassificationPanel from './components/ClassificationPanel';
import MorphometricsPanel from './components/MorphometricsPanel';
import ExplainabilityPanel from './components/ExplainabilityPanel';
import PerformancePanel from './components/PerformancePanel';
import ClinicalDisclaimer from './components/ClinicalDisclaimer';
import SignalAnalysisTab from './components/SignalAnalysisTab';
import { RefreshCw, AlertCircle, Radio, BrainCircuit, ShieldAlert } from 'lucide-react';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [ssData, setSsData] = useState(null);
  const [error, setError] = useState(null);
  const [ssError, setSsError] = useState(null);
  const [modalImage, setModalImage] = useState(null);

  // Navigation Tabs
  const [mainTab, setMainTab] = useState('segmentation'); // 'segmentation' | 'signal_analysis'
  const [overlayTab, setOverlayTab] = useState('primary'); // 'primary' | 'secondary'

  const handleAnalyze = async (file, modelName = 'brats') => {
    setLoading(true);
    setError(null);
    setSsError(null);
    setPredictionData(null);
    setSsData(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelName);

    const formDataSs = new FormData();
    formDataSs.append('file', file);

    try {
      // Execute both requests concurrently
      const [predRes, ssRes] = await Promise.allSettled([
        fetch('http://localhost:8000/predict', { method: 'POST', body: formData }),
        fetch('http://localhost:8000/analyze-signal', { method: 'POST', body: formDataSs })
      ]);

      // Process Prediction Result
      if (predRes.status === 'fulfilled' && predRes.value.ok) {
        const predJson = await predRes.value.json();
        if (predJson.success) {
          setPredictionData(predJson);
        } else {
          setError(predJson.detail || 'Segmentation failed. Please try another image.');
        }
      } else {
        const isNetworkErr = predRes.status === 'rejected' || !predRes.value;
        const errDetail = isNetworkErr
          ? 'Unable to connect to the SUSHRUTA backend.'
          : (await predRes.value.json().catch(() => ({}))).detail || 'Segmentation failed. Please try another image.';
        setError(errDetail);
      }

      // Process Signals & Systems Result
      if (ssRes.status === 'fulfilled' && ssRes.value.ok) {
        const ssJson = await ssRes.value.json();
        if (ssJson.success) {
          setSsData(ssJson);
        } else {
          setSsError(ssJson.detail || 'Signal analysis failed. Please try another image.');
        }
      } else {
        const isNetworkErr = ssRes.status === 'rejected' || !ssRes.value;
        const ssErrDetail = isNetworkErr
          ? 'Unable to connect to the SUSHRUTA backend.'
          : (await ssRes.value.json().catch(() => ({}))).detail || 'Signal analysis failed.';
        setSsError(ssErrDetail);
      }

    } catch (err) {
      setError('Unable to connect to the SUSHRUTA backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPredictionData(null);
    setSsData(null);
    setError(null);
    setSsError(null);
    setModalImage(null);
  };

  // Pipeline verdict first: figshare verdict comes from segmentation (the BraTS
  // classifier sub-object stays honest but is out-of-domain on T1-CE).
  const isTumorDetected = predictionData?.tumor_detected ?? predictionData?.classification?.tumor_detected ?? false;

  return (
    <div className="app-container">
      <Header />

      <main>
        {/* Section 1: Upload / Input */}
        <UploadZone onAnalyze={handleAnalyze} isLoading={loading} />

        {/* Global Error Banner */}
        {error && (
          <div style={{
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
            color: 'var(--accent-rose)',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontSize: '0.9rem'
          }}>
            <AlertCircle size={20} />
            <div>
              <strong>Prediction Error: </strong> {error}
            </div>
          </div>
        )}

        {/* Major Section Tab Bar */}
        {(predictionData || ssData) && (
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1.5rem',
            borderBottom: '1px solid var(--border-color)',
            paddingBottom: '0.75rem',
            flexWrap: 'wrap',
            gap: '1rem'
          }}>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn-secondary"
                style={{
                  background: mainTab === 'segmentation' ? 'var(--accent-cyan)' : 'var(--bg-card)',
                  color: mainTab === 'segmentation' ? '#090d16' : 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  padding: '0.6rem 1.25rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
                onClick={() => setMainTab('segmentation')}
              >
                <BrainCircuit size={18} />
                Segmentation & XAI
              </button>

              <button
                className="btn-secondary"
                style={{
                  background: mainTab === 'signal_analysis' ? 'var(--accent-cyan)' : 'var(--bg-card)',
                  color: mainTab === 'signal_analysis' ? '#090d16' : 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  padding: '0.6rem 1.25rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
                onClick={() => setMainTab('signal_analysis')}
              >
                <Radio size={18} />
                Signal & Image Analysis
              </button>
            </div>

            <button className="btn-secondary" onClick={handleReset}>
              <RefreshCw size={16} style={{ marginRight: '0.3rem' }} />
              Analyze Another Image
            </button>
          </div>
        )}

        {/* TAB 1: SEGMENTATION & XAI */}
        {mainTab === 'segmentation' && predictionData && (
          <div>
            {/* STAGE 1: CLASSIFICATION & DECISION EXPLANATION */}
            <ClassificationPanel
              classification={predictionData.classification}
              classifierLayercamUrl={predictionData.outputs?.classifier_layercam}
              onOpenModal={(imgInfo) => setModalImage(imgInfo)}
            />

            {/* STAGE 2: U-NET SEGMENTATION (ONLY WHEN TUMOR IS DETECTED) */}
            {isTumorDetected ? (
              <>
                {/* Overlay View Switcher */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '1.25rem'
                }}>
                  <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      Stage 2: U-Net Tumor Segmentation & Detailed XAI
                    </h2>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Pixel-wise tumor delineation (U-Net) and decoder feature attributions
                    </p>
                  </div>

                  <div style={{ background: 'var(--bg-card)', padding: '4px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', display: 'flex', gap: '4px' }}>
                    <button
                      className="btn-secondary"
                      style={{
                        background: overlayTab === 'primary' ? 'var(--accent-cyan)' : 'transparent',
                        color: overlayTab === 'primary' ? '#090d16' : 'var(--text-secondary)',
                        border: 'none',
                        fontWeight: 600,
                        fontSize: '0.8rem'
                      }}
                      onClick={() => setOverlayTab('primary')}
                    >
                      Primary Overlays
                    </button>
                    <button
                      className="btn-secondary"
                      style={{
                        background: overlayTab === 'secondary' ? 'var(--accent-cyan)' : 'transparent',
                        color: overlayTab === 'secondary' ? '#090d16' : 'var(--text-secondary)',
                        border: 'none',
                        fontWeight: 600,
                        fontSize: '0.8rem'
                      }}
                      onClick={() => setOverlayTab('secondary')}
                    >
                      Standalone Maps
                    </button>
                  </div>
                </div>

                {/* RESULTS VISUAL CARDS */}
                {overlayTab === 'primary' ? (
                  <div className="results-grid-4">
                    <VisualCard
                      title="Original Preprocessed MRI"
                      subtitle="2D FLAIR slice normalized to [0.0, 1.0]"
                      imageUrl={predictionData.outputs?.preprocessed_mri}
                      onClick={() => setModalImage({
                        title: 'Original Preprocessed MRI',
                        subtitle: '2D Brain MRI FLAIR slice after canonical float scaling [0, 1] and 256x256 resizing.',
                        url: predictionData.outputs?.preprocessed_mri
                      })}
                    />

                    <VisualCard
                      title="Tumor Segmentation"
                      subtitle="Binary mask overlay on MRI (Yellow Box)"
                      imageUrl={predictionData.outputs?.overlay}
                      onClick={() => setModalImage({
                        title: 'Tumor Segmentation Overlay',
                        subtitle: 'Predicted tumor region (green semi-transparent mask) with overall bounding box outline.',
                        url: predictionData.outputs?.overlay
                      })}
                    />

                    <VisualCard
                      title="Segmentation LayerCAM"
                      subtitle="MRI + U-Net feature attribution overlay"
                      imageUrl={predictionData.outputs?.layercam_overlay}
                      onClick={() => setModalImage({
                        title: 'U-Net LayerCAM Feature Attribution Overlay',
                        subtitle: 'LayerCAM activation heatmap (Jiang et al., 2021) backpropagated from decoder layer up4.',
                        url: predictionData.outputs?.layercam_overlay
                      })}
                    />

                    <VisualCard
                      title="Combined Analysis"
                      subtitle="LayerCAM + Cyan Tumor Contour boundary"
                      imageUrl={predictionData.outputs?.combined_overlay}
                      onClick={() => setModalImage({
                        title: 'Combined Diagnostic Analysis',
                        subtitle: 'Simultaneous overlay comparing tumor segmentation boundary (cyan contour) against LayerCAM attribution gradients.',
                        url: predictionData.outputs?.combined_overlay
                      })}
                    />
                  </div>
                ) : (
                  <div className="results-grid-4">
                    <VisualCard
                      title="Binary Tumor Mask"
                      subtitle="Thresholded decision mask {0, 1}"
                      imageUrl={predictionData.outputs?.tumor_mask}
                      onClick={() => setModalImage({
                        title: 'Binary Tumor Mask',
                        subtitle: '256x256 binary mask where background = 0 (black) and tumor = 1 (white).',
                        url: predictionData.outputs?.tumor_mask
                      })}
                    />

                    <VisualCard
                      title="Probability Heatmap"
                      subtitle="Continuous U-Net sigmoid output P ∈ [0, 1]"
                      imageUrl={predictionData.outputs?.probability_map}
                      onClick={() => setModalImage({
                        title: 'Continuous Probability Heatmap',
                        subtitle: 'U-Net sigmoid probability map rendered with JET colormap.',
                        url: predictionData.outputs?.probability_map
                      })}
                    />

                    <VisualCard
                      title="LayerCAM Heatmap"
                      subtitle="Standalone attribution activation map"
                      imageUrl={predictionData.outputs?.layercam}
                      onClick={() => setModalImage({
                        title: 'Standalone LayerCAM Heatmap',
                        subtitle: 'Normalized LayerCAM attribution map L ∈ [0, 1].',
                        url: predictionData.outputs?.layercam
                      })}
                    />
                  </div>
                )}

                {/* QUANTITATIVE MORPHOMETRICS */}
                <MorphometricsPanel
                  measurements={predictionData.measurements}
                  confidence={predictionData.segmentation_confidence}
                  tumorDetected={predictionData.tumor_detected}
                />

                {/* EXPLAINABILITY DETAILS */}
                <ExplainabilityPanel
                  explainability={predictionData.explainability}
                />
              </>
            ) : (
              <div style={{
                background: 'rgba(15, 23, 42, 0.5)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: 'var(--radius-md)',
                padding: '1.5rem',
                marginBottom: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}>
                <ShieldAlert size={28} color="var(--accent-emerald)" style={{ flexShrink: 0 }} />
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--accent-emerald)', marginBottom: '0.25rem' }}>
                    Stage 2 U-Net Tumor Segmentation Skipped
                  </div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                    No tumor region was identified above the classifier's decision threshold ({(predictionData?.classification?.decision_threshold * 100).toFixed(0)}%). Pixel-wise U-Net segmentation and tumor morphometrics were omitted to prevent false positive tumor reporting.
                  </div>
                </div>
              </div>
            )}

            {/* PERFORMANCE BENCHMARKS */}
            <PerformancePanel
              inference={predictionData.inference}
            />
          </div>
        )}

        {/* TAB 2: SIGNAL & IMAGE ANALYSIS */}
        {mainTab === 'signal_analysis' && (
          <SignalAnalysisTab
            ssData={ssData}
            isLoading={loading}
            error={ssError}
            onImageClick={(imgInfo) => setModalImage(imgInfo)}
          />
        )}

        {/* CLINICAL DISCLAIMER */}
        <ClinicalDisclaimer
          text={predictionData?.disclaimer}
        />
      </main>

      {/* Lightbox Modal */}
      {modalImage && (
        <ImageModal
          title={modalImage.title}
          subtitle={modalImage.subtitle}
          imageUrl={modalImage.url}
          onClose={() => setModalImage(null)}
        />
      )}
    </div>
  );
}
