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
import { RefreshCw, AlertCircle, Radio, BrainCircuit, Activity, CheckCircle2, AlertTriangle } from 'lucide-react';

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

  const handleAnalyze = async (file) => {
    setLoading(true);
    setError(null);
    setSsError(null);
    setPredictionData(null);
    setSsData(null);

    const formData = new FormData();
    formData.append('file', file);

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

  // Quantitative metrics & model agreement calculations
  const segAreaPixels = predictionData?.measurements?.area_pixels ?? 0;
  const segAreaPct = predictionData?.measurements?.area_percentage ?? 0.0;
  const segHasTumor = segAreaPixels > 0;
  const segOutcomeText = segHasTumor
    ? 'Whole-tumor pixels segmented'
    : 'No whole-tumor pixels segmented';

  const clsTumorDetected = predictionData?.classification?.tumor_detected ?? false;
  const clsProb = predictionData?.classification?.tumor_probability ?? 0.0;
  const clsThreshold = predictionData?.classification?.decision_threshold ?? 0.50;

  // Model agreement: comparison between auxiliary classifier and U-Net mask
  const isModelAgreement = (clsTumorDetected === segHasTumor);
  const agreementLabel = isModelAgreement ? 'Models Concordant' : 'Model Discrepancy';
  const agreementDetail = isModelAgreement
    ? (clsTumorDetected
        ? 'Both models indicate tumor presence (Screening P ≥ 50% and U-Net segmented whole-tumor pixels)'
        : 'Neither model indicates tumor presence (Screening P < 50% and U-Net segmented 0 pixels above threshold τ = 0.50)')
    : (clsTumorDetected
        ? 'Model Discrepancy: Auxiliary 2D CNN classifier indicated tumor (P ≥ 50%), but U-Net segmented 0 pixels above threshold τ = 0.50'
        : 'Model Discrepancy: Auxiliary 2D CNN classifier was below threshold (P < 50%), but U-Net segmented whole-tumor pixels above threshold τ = 0.50');

  const unetLatency = predictionData?.inference?.segmentation_inference_time_ms ?? predictionData?.inference?.time_ms;
  const totalLatency = predictionData?.inference?.total_request_time_ms;

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
            {/* PREDICTION SUMMARY & MODEL AGREEMENT (SECTION C) */}
            <div className="card" style={{
              marginBottom: '1.25rem',
              padding: '1.25rem',
              background: 'rgba(18, 24, 39, 0.85)',
              border: '1px solid var(--border-color)',
              boxShadow: 'var(--shadow-md)'
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '0.75rem',
                marginBottom: '1rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
                paddingBottom: '0.75rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <Activity size={20} color="var(--accent-cyan)" />
                  <span style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                    Inference Analysis Summary & Model Agreement
                  </span>
                </div>

                {/* Agreement Badge with explicit non-consensus disclaimer */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                  <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '4px 10px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    background: isModelAgreement ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                    color: isModelAgreement ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                    border: isModelAgreement ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(245, 158, 11, 0.3)'
                  }}>
                    {isModelAgreement ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                    <span>{agreementLabel}</span>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    (Model-output agreement only, not diagnostic consensus)
                  </span>
                </div>
              </div>

              {/* 4-Stat Metric Summary Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: '1rem',
                marginBottom: '1rem'
              }}>
                {/* Stat 1: U-Net Segmentation Outcome */}
                <div style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(9, 13, 22, 0.6)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.25rem' }}>
                    U-Net Segmentation Outcome
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: segHasTumor ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                    {segOutcomeText}
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    {segAreaPixels.toLocaleString()} px ({Number(segAreaPct).toFixed(2)}% of 256×256)
                  </div>
                </div>

                {/* Stat 2: Auxiliary Screening Classifier */}
                <div style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(9, 13, 22, 0.6)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Auxiliary Screening Classifier (2D CNN)
                  </div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: clsTumorDetected ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                    {clsTumorDetected ? 'Tumor Indicated' : 'No Tumor Indicated'}
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    P = {(clsProb * 100).toFixed(1)}% (Threshold: {(clsThreshold * 100).toFixed(0)}%)
                  </div>
                </div>

                {/* Stat 3: Model Output Agreement */}
                <div style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(9, 13, 22, 0.6)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Model Agreement Details
                  </div>
                  <div style={{ fontSize: '0.825rem', fontWeight: 500, color: 'var(--text-primary)', lineHeight: 1.35 }}>
                    {agreementDetail}
                  </div>
                </div>

                {/* Stat 4: Processing Latency */}
                <div style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(9, 13, 22, 0.6)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Processing Latencies
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                    U-Net: <strong>{unetLatency ? unetLatency.toFixed(1) + ' ms' : 'N/A'}</strong>
                  </div>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    Total Request: <strong>{totalLatency ? totalLatency.toFixed(1) + ' ms' : 'N/A'}</strong>
                  </div>
                </div>
              </div>

              {/* Slice Context Explanatory Note */}
              <div style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                lineHeight: 1.45,
                borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                paddingTop: '0.5rem',
                fontStyle: 'italic'
              }}>
                * Model Output Interpretation: This analysis represents the algorithmic model output for this specific 2D slice. An empty segmentation mask reflects zero pixels exceeding the threshold (τ = 0.50) on this slice, not clinical confirmation of tumor absence.
              </div>
            </div>

            {/* COMPACT AUXILIARY CLASSIFICATION SECTION */}
            <ClassificationPanel
              classification={predictionData.classification}
              classifierLayercamUrl={predictionData.outputs?.classifier_layercam}
              onOpenModal={(imgInfo) => setModalImage(imgInfo)}
            />

            {/* UNIFIED RESULTS WORKSPACE (UNCONDITIONALLY DISPLAYED) */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1rem',
              flexWrap: 'wrap',
              gap: '0.75rem'
            }}>
              <div>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  MRI Segmentation & Visual Interpretability
                </h2>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  Unified multi-view workspace displaying slice input, whole-tumor delineation, and feature attributions
                </p>
              </div>

              <div style={{
                background: 'var(--bg-card)',
                padding: '4px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-color)',
                display: 'flex',
                gap: '4px'
              }}>
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
                  Primary Viewport (Core 4 Views)
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
                  Diagnostic Heatmaps & Secondary Views
                </button>
              </div>
            </div>

            {/* PRIMARY VIEWPORT VS SECONDARY VIEWS */}
            {overlayTab === 'primary' ? (
              <div className="results-grid-4">
                {/* View 1: Preprocessed MRI */}
                <VisualCard
                  title="1. Preprocessed MRI Slice"
                  subtitle="Canonical float scaling [0, 1] & 256×256 matrix"
                  imageUrl={predictionData.outputs?.preprocessed_mri}
                  onClick={() => setModalImage({
                    title: 'Original Preprocessed MRI',
                    subtitle: '2D Brain MRI slice after canonical float scaling [0.0, 1.0] and 256×256 resizing.',
                    url: predictionData.outputs?.preprocessed_mri
                  })}
                />

                {/* View 2: Binary Whole-Tumor Segmentation Mask */}
                <VisualCard
                  title="2. Binary Whole-Tumor Mask"
                  subtitle={segHasTumor ? `${segAreaPixels.toLocaleString()} px segmented (τ = 0.50)` : "0 pixels segmented (all-background mask)"}
                  imageUrl={predictionData.outputs?.tumor_mask}
                  onClick={() => setModalImage({
                    title: 'Binary Whole-Tumor Segmentation Mask',
                    subtitle: '256×256 binary mask where background = 0 (black) and predicted whole-tumor = 1 (white).',
                    url: predictionData.outputs?.tumor_mask
                  })}
                />

                {/* View 3: MRI + Segmentation Overlay */}
                <VisualCard
                  title="3. MRI + Tumor Overlay"
                  subtitle={segHasTumor ? "Predicted whole-tumor region (green) with bounding box" : "MRI slice with no tumor overlay (0 px segmented)"}
                  imageUrl={predictionData.outputs?.overlay}
                  onClick={() => setModalImage({
                    title: 'MRI + Tumor Segmentation Overlay',
                    subtitle: segHasTumor
                      ? 'Predicted whole-tumor region (green semi-transparent mask) with overall bounding box outline.'
                      : 'Preprocessed MRI slice without tumor overlay because 0 tumor pixels were segmented.',
                    url: predictionData.outputs?.overlay
                  })}
                />

                {/* View 4: U-Net LayerCAM Attribution */}
                <VisualCard
                  title="4. U-Net LayerCAM Attribution"
                  subtitle={predictionData.outputs?.layercam_overlay ? "MRI + decoder layer up4 feature attribution" : "Unavailable (0 tumor pixels segmented)"}
                  imageUrl={predictionData.outputs?.layercam_overlay}
                  badge="LayerCAM Attribution Unavailable"
                  unavailableMessage={
                    predictionData.explainability?.interpretation ||
                    "Tumor-targeted LayerCAM attribution requires predicted tumor pixels to backpropagate gradients. Because 0 tumor pixels were segmented above threshold τ = 0.50, attribution is unavailable for this slice."
                  }
                  onClick={() => setModalImage({
                    title: 'U-Net LayerCAM Feature Attribution Overlay',
                    subtitle: 'LayerCAM activation heatmap (Jiang et al., 2021) backpropagated from decoder layer up4.',
                    url: predictionData.outputs?.layercam_overlay
                  })}
                />
              </div>
            ) : (
              <div className="results-grid-4">
                {/* Secondary 1: Probability Heatmap */}
                <VisualCard
                  title="Continuous Probability Heatmap"
                  subtitle={
                    segHasTumor
                      ? "Sigmoid probability map P ∈ [0.0, 1.0] (JET)"
                      : "Continuous probability map (All pixels below τ = 0.50 decision threshold)"
                  }
                  imageUrl={predictionData.outputs?.probability_map}
                  onClick={() => setModalImage({
                    title: 'Continuous Probability Heatmap',
                    subtitle: segHasTumor
                      ? 'Continuous pixel-wise sigmoid probability map rendered with JET colormap.'
                      : 'Continuous pixel-wise sigmoid probability map. Note: All pixels are below the decision threshold (τ = 0.50); no whole-tumor region is segmented.',
                    url: predictionData.outputs?.probability_map
                  })}
                />

                {/* Secondary 2: Combined Contour / Overlay */}
                <VisualCard
                  title="Combined Overlay & Contour"
                  subtitle={predictionData.outputs?.combined_overlay ? "LayerCAM + Cyan Tumor Contour boundary" : "Unavailable (No tumor contour/attribution)"}
                  imageUrl={predictionData.outputs?.combined_overlay}
                  badge="Combined Contour Unavailable"
                  unavailableMessage="Combined overlay requires both tumor contour and LayerCAM attribution gradients, which are absent for an empty mask."
                  onClick={() => setModalImage({
                    title: 'Combined Diagnostic Analysis',
                    subtitle: 'Simultaneous overlay comparing tumor segmentation boundary (cyan contour) against LayerCAM attribution gradients.',
                    url: predictionData.outputs?.combined_overlay
                  })}
                />

                {/* Secondary 3: Classifier LayerCAM Overlay */}
                <VisualCard
                  title="Classifier LayerCAM Overlay"
                  subtitle={
                    clsTumorDetected
                      ? `Auxiliary 2D CNN screening focus region (${predictionData.classification?.xai?.strongest_region_location && predictionData.classification?.xai?.strongest_region_location !== 'none' ? predictionData.classification?.xai?.strongest_region_location : 'focal'})`
                      : `Auxiliary 2D CNN sub-threshold feature trace (P = ${(clsProb * 100).toFixed(1)}% < 50%)`
                  }
                  imageUrl={predictionData.outputs?.classifier_layercam}
                  onClick={() => setModalImage({
                    title: clsTumorDetected ? 'Classifier LayerCAM Heatmap Overlay' : 'Sub-Threshold Classifier Attribution Overlay',
                    subtitle: clsTumorDetected
                      ? 'Feature attribution for auxiliary 2D CNN classifier indicating image regions driving positive screening.'
                      : 'Sub-threshold feature attribution map for auxiliary 2D CNN classifier. No focal lesion confirmed above decision threshold.',
                    url: predictionData.outputs?.classifier_layercam
                  })}
                />

                {/* Secondary 4: Standalone LayerCAM Heatmap */}
                <VisualCard
                  title="Standalone LayerCAM Heatmap"
                  subtitle={predictionData.outputs?.layercam ? "Normalized attribution heatmap L ∈ [0.0, 1.0]" : "Unavailable (0 tumor pixels segmented)"}
                  imageUrl={predictionData.outputs?.layercam}
                  badge="LayerCAM Heatmap Unavailable"
                  unavailableMessage="Standalone LayerCAM heatmap is unavailable because 0 tumor pixels were segmented."
                  onClick={() => setModalImage({
                    title: 'Standalone LayerCAM Heatmap',
                    subtitle: 'Normalized LayerCAM attribution map L ∈ [0.0, 1.0].',
                    url: predictionData.outputs?.layercam
                  })}
                />
              </div>
            )}

            {/* QUANTITATIVE MORPHOMETRICS (SECTION D) */}
            <MorphometricsPanel
              measurements={predictionData.measurements}
              confidence={predictionData.segmentation_confidence}
              tumorDetected={segHasTumor}
            />

            {/* EXPLAINABILITY DETAILS */}
            <ExplainabilityPanel
              explainability={predictionData.explainability}
            />

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
