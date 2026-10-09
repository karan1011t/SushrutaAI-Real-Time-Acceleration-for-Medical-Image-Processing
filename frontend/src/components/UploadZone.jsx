import React, { useState, useRef } from 'react';
import { UploadCloud, FileImage, Trash2, Play, Loader2, AlertCircle, Info } from 'lucide-react';

export default function UploadZone({ onAnalyze, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    setErrorMsg(null);
    if (!file) return;

    const validExts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.nii', '.gz'];
    const fileName = file.name.toLowerCase();
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const isValid = validExts.some(valid => fileName.endsWith(valid));

    if (!isValid) {
      setErrorMsg(`Unsupported file type '${file.name}'. Please select a valid PNG, JPG, BMP, TIFF, or NIfTI (.nii/.nii.gz) Brain MRI image.`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Read image dimensions
    const img = new Image();
    img.src = url;
    img.onload = () => {
      setFileMeta({
        name: file.name,
        size: (file.size / 1024).toFixed(1) + ' KB',
        dimensions: `${img.width} × ${img.height} px`,
        format: ext.replace('.', '').toUpperCase()
      });
    };
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFileMeta(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleStartAnalysis = () => {
    if (selectedFile && !isLoading) {
      onAnalyze(selectedFile);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      <div className="card-title">
        <FileImage size={20} color="var(--accent-cyan)" />
        <span>MRI Input Selection</span>
      </div>
      <p className="card-subtitle" style={{ marginBottom: '0.75rem' }}>
        Upload a compatible 2D Brain MRI slice for U-Net segmentation and LayerCAM explainability analysis.
      </p>

      {/* Domain Scope Academic Notice */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        background: 'rgba(56, 189, 248, 0.06)',
        border: '1px solid rgba(56, 189, 248, 0.2)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.6rem 0.85rem',
        marginBottom: '1.25rem',
        fontSize: '0.78rem',
        color: 'var(--text-secondary)',
        lineHeight: '1.4'
      }}>
        <Info size={16} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Validation Scope:</strong>{' '}
          Validated for BraTS 2021 2D FLAIR MRI slices. Predictions on non-FLAIR sequences or external/web JPEG images may exhibit out-of-distribution artifacts. NIfTI inputs are processed through the existing 2D slice pathway.
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => handleFileChange(e.target.files[0])}
        accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
        style={{ display: 'none' }}
      />

      {!previewUrl ? (
        <div
          className={`dropzone ${isDragging ? 'active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadCloud className="upload-icon" />
          <div>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              Click to browse
            </span>{' '}
            or drag and drop your MRI file here
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Supported File Formats: <code style={{ color: 'var(--accent-cyan)' }}>PNG • JPG • JPEG • BMP • TIFF • NIfTI</code>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Model Evaluation Domain: BraTS 2021 2D FLAIR axial MRI slices
          </div>
        </div>
      ) : (
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '1.25rem' }}>
          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ width: '120px', height: '120px', borderRadius: 'var(--radius-sm)', overflow: 'hidden', background: '#000', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img src={previewUrl} alt="MRI Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            </div>

            <div style={{ flex: 1, minWidth: '220px' }}>
              <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '0.5rem', wordBreak: 'break-all' }}>
                {fileMeta?.name}
              </div>
              <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                <span>Dimensions: <strong style={{ color: 'var(--text-primary)' }}>{fileMeta?.dimensions}</strong></span>
                <span>Size: <strong style={{ color: 'var(--text-primary)' }}>{fileMeta?.size}</strong></span>
                <span>Format: <strong style={{ color: 'var(--accent-cyan)' }}>{fileMeta?.format}</strong></span>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <button
                  className="btn-primary"
                  onClick={handleStartAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      <div>
                        <div style={{ fontWeight: 600 }}>Analyzing MRI...</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.8, fontWeight: 400 }}>Processing segmentation and explainability...</div>
                      </div>
                    </>
                  ) : (
                    <>
                      <Play size={18} />
                      <span>Analyze MRI</span>
                    </>
                  )}
                </button>

                <button className="btn-secondary" onClick={handleClear} disabled={isLoading}>
                  <Trash2 size={16} style={{ marginRight: '0.3rem', verticalAlign: 'middle' }} />
                  Remove
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.3)', color: 'var(--accent-rose)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={18} />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}
