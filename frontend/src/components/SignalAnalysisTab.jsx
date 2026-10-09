import React from 'react';
import VisualCard from './VisualCard';
import { Activity, Radio, Cpu, Layers, Zap, Info, Clock, Sliders } from 'lucide-react';

export default function SignalAnalysisTab({ ssData, isLoading, error, onImageClick }) {
  if (isLoading) {
    return (
      <div style={{
        padding: '3rem',
        textAlign: 'center',
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-color)',
        color: 'var(--text-secondary)'
      }}>
        <Activity size={32} className="spin-animation" style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }} />
        <h3 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Processing 2D Discrete Spatial Signal Analysis...</h3>
        <p style={{ fontSize: '0.9rem' }}>Computing 2D FFT spectrum, spatial filters, Sobel/Canny edges, binary morphology, and SNR metrics.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem',
        background: 'rgba(244, 63, 94, 0.1)',
        border: '1px solid rgba(244, 63, 94, 0.3)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--accent-rose)'
      }}>
        <strong>Signal Analysis Error: </strong> {error}
      </div>
    );
  }

  if (!ssData) {
    return (
      <div style={{
        padding: '3rem',
        textAlign: 'center',
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-color)',
        color: 'var(--text-secondary)'
      }}>
        <Radio size={36} style={{ color: 'var(--accent-cyan)', marginBottom: '1rem', opacity: 0.6 }} />
        <h3 style={{ color: 'var(--text-primary)' }}>No Signal Analysis Data Available</h3>
        <p style={{ fontSize: '0.9rem' }}>Upload an MRI slice to view full Signals & Systems / Image Processing metrics.</p>
      </div>
    );
  }

  const {
    signal,
    histogram,
    filtering,
    frequency,
    edges,
    morphology,
    connected_components: cc,
    snr_estimate: snr,
    timings,
    outputs
  } = ssData;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* HEADER BANNER */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(0, 229, 255, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
        border: '1px solid rgba(0, 229, 255, 0.2)',
        borderRadius: 'var(--radius-md)',
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Radio size={20} style={{ color: 'var(--accent-cyan)' }} />
            Signals & Systems / Image Processing Analysis Layer
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0 0' }}>
            Analytical spatial and frequency domain exploration of the 2D MRI discrete spatial signal I[m,n].
            <strong style={{ color: 'var(--accent-cyan)', marginLeft: '0.4rem' }}>
              (Educational layer — does NOT modify production U-Net inference).
            </strong>
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{
            background: 'rgba(9, 13, 22, 0.6)',
            padding: '0.5rem 0.85rem',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-color)',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)'
          }}>
            <Clock size={14} style={{ marginRight: '0.3rem', color: 'var(--accent-cyan)', verticalAlign: 'middle' }} />
            SS Analysis Time: <strong style={{ color: 'var(--text-primary)' }}>{timings?.total_signal_analysis_time_ms} ms</strong>
          </div>
        </div>
      </div>

      {/* SECTION 1: SIGNAL STATISTICS & HISTOGRAM */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>

        {/* Signal Stats Panel */}
        <div style={{
          background: 'var(--bg-card)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-color)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Activity size={18} style={{ color: 'var(--accent-cyan)' }} />
              MRI as a 2D Discrete Spatial Signal I[m,n]
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Grid Dimensions</span>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  {signal?.width} × {signal?.height}
                </div>
              </div>

              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Total Spatial Pixels</span>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  {signal?.pixels?.toLocaleString()}
                </div>
              </div>

              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Mean Intensity (μ)</span>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  {signal?.mean}
                </div>
              </div>

              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Std Dev (σ)</span>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  {signal?.std}
                </div>
              </div>

              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Dynamic Range</span>
                <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                  {signal?.dynamic_range}
                </div>
              </div>

              <div style={{ background: 'var(--bg-card-hover)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Estimated SNR</span>
                <div style={{ fontWeight: 700, color: 'var(--accent-cyan)', fontSize: '0.95rem' }}>
                  {snr?.tissue_snr_db} dB <span style={{ fontSize: '0.75rem', fontWeight: 400, color: 'var(--text-secondary)' }}>({snr?.tissue_snr_linear})</span>
                </div>
              </div>
            </div>
          </div>

          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.02)', padding: '0.6rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)', margin: 0 }}>
            <Info size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
            Each MRI slice is represented as a two-dimensional array of sampled intensity values I[m,n].
          </p>
        </div>

        {/* Intensity Histogram Visual Card */}
        <VisualCard
          title="Concept 2: Intensity Distribution"
          subtitle={`256-Bin Intensity Histogram (μ = ${histogram?.mean}, σ = ${histogram?.std})`}
          imageUrl={outputs?.histogram}
          onClick={() => onImageClick && onImageClick({
            title: '256-Bin MRI Intensity Histogram',
            subtitle: 'Distribution of sampled pixel intensities across normalized [0, 1] range.',
            url: outputs?.histogram
          })}
        />
      </div>

      {/* SECTION 2: SPATIAL FILTERING & 2D FFT */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.25rem' }}>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <VisualCard
            title="Concept 3: Spatial Filtering Comparison"
            subtitle={`Original vs Gaussian 5x5 (σ=1.5) vs Median 5x5 (${timings?.filtering_time_ms} ms)`}
            imageUrl={outputs?.filter_comparison}
            onClick={() => onImageClick && onImageClick({
              title: 'Spatial Domain Filtering Experiments',
              subtitle: 'Comparison between Gaussian smoothing (5x5) and non-linear Median filtering (5x5).',
              url: outputs?.filter_comparison
            })}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '0.5rem 0.75rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', margin: 0 }}>
            <Sliders size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
            Spatial filters modify a pixel using information from its local neighborhood. Gaussian filtering provides smoothing, while median filtering is useful for reducing impulse-like noise while preserving edges.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <VisualCard
            title="Concept 4: 2D Discrete Fourier Transform (FFT)"
            subtitle={`Original I[m,n] | Magnitude |F[u,v]| | Log Spectrum log(1+|F|) (${timings?.fft_time_ms} ms)`}
            imageUrl={outputs?.fft_analysis}
            onClick={() => onImageClick && onImageClick({
              title: '2D Discrete Fourier Transform Analysis',
              subtitle: 'Spatial frequency spectrum decomposition. Low frequency ratio = ' + ((frequency?.low_frequency_energy_ratio ?? 0) * 100).toFixed(1) + '%.',
              url: outputs?.fft_analysis
            })}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '0.5rem 0.75rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', margin: 0 }}>
            <Zap size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
            The Fourier transform represents the MRI in the spatial-frequency domain. Low spatial frequencies correspond mainly to slowly varying intensity structures, while higher spatial frequencies correspond to finer details and rapid intensity changes.
          </p>
        </div>

      </div>

      {/* SECTION 3: EDGE DETECTION & MATHEMATICAL MORPHOLOGY */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.25rem' }}>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <VisualCard
            title="Concept 5: Spatial Edge Detection"
            subtitle={`Sobel Gradient (|G| > 0.3: ${edges?.sobel_edge_pixels} px) | Canny Hysteresis (${edges?.canny_edge_pixels} px)`}
            imageUrl={outputs?.edge_detection}
            onClick={() => onImageClick && onImageClick({
              title: 'Spatial Edge Detection Comparison',
              subtitle: 'First-order Sobel derivative magnitude vs multi-stage Canny edge detector.',
              url: outputs?.edge_detection
            })}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '0.5rem 0.75rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', margin: 0 }}>
            <Layers size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
            Edges correspond to rapid spatial intensity changes and therefore contain high-frequency structural information.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <VisualCard
            title="Concepts 6 & 7: Thresholding & Binary Morphology"
            subtitle={`Probability Threshold P ≥ 0.5 → Erosion | Dilation | Opening | Closing (${timings?.morphology_time_ms} ms)`}
            imageUrl={outputs?.morphology_analysis}
            onClick={() => onImageClick && onImageClick({
              title: 'Mathematical Morphology Progression',
              subtitle: 'Ablation demonstration of 5x5 elliptical morphological structuring operations on binary decision mask.',
              url: outputs?.morphology_analysis
            })}
          />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '0.5rem 0.75rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', margin: 0 }}>
            <Cpu size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
            Thresholding converts the continuous model probability output into a binary segmentation signal M(x,y). Morphological operations refine binary spatial shapes using set operations.
          </p>
        </div>

      </div>

      {/* SECTION 4: CONNECTED COMPONENTS & ESTIMATED SNR BREAKDOWN */}
      <div style={{
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-color)',
        padding: '1.25rem'
      }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Layers size={18} style={{ color: 'var(--accent-cyan)' }} />
          Concept 8 & 9: Connected Components & Signal Quality Metrics
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
          <div style={{ background: 'var(--bg-card-hover)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Connected Components</span>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
              {cc?.component_count}
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Discrete spatial clusters</span>
          </div>

          <div style={{ background: 'var(--bg-card-hover)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Largest Component Area</span>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-cyan)', marginTop: '0.2rem' }}>
              {cc?.largest_component_area_pixels} px
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Dominant lesion foreground</span>
          </div>

          <div style={{ background: 'var(--bg-card-hover)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Tissue SNR Estimate</span>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
              {snr?.tissue_snr_db} dB
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Linear = {snr?.tissue_snr_linear}</span>
          </div>

          <div style={{ background: 'var(--bg-card-hover)', padding: '0.8rem 1rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Global SNR Estimate</span>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.2rem' }}>
              {snr?.global_snr_db} dB
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Linear = {snr?.global_snr_linear}</span>
          </div>
        </div>

        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '1rem', marginBottom: 0 }}>
          <Info size={14} style={{ color: 'var(--accent-cyan)', marginRight: '0.3rem', verticalAlign: 'middle' }} />
          Connected-component analysis treats connected foreground pixels as separate regions and allows us to quantify segmented structures.
          Tissue SNR estimation method: <em>{snr?.estimation_method}</em> (* Academic image-quality estimate; not a clinical acquisition SNR measurement).
        </p>
      </div>

    </div>
  );
}
