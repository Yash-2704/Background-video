import React, { useState, useEffect, useRef } from 'react'
import { uploadImage, submitAnimation, API_BASE } from '../api/client.js'

export default function AnimateForm({ onAnimateSuccess }) {
  const [file, setFile] = useState(null)
  const [localPreviewUrl, setLocalPreviewUrl] = useState(null)
  const [stage, setStage] = useState('upload')
  const [uploadResult, setUploadResult] = useState(null)
  const [animationPrompt, setAnimationPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fileInputRef = useRef(null)

  useEffect(() => {
    return () => {
      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl)
      }
    }
  }, [localPreviewUrl])

  function handleFileChange(e) {
    const selected = e.target.files?.[0]
    if (!selected) return
    if (localPreviewUrl) {
      URL.revokeObjectURL(localPreviewUrl)
    }
    setFile(selected)
    setLocalPreviewUrl(URL.createObjectURL(selected))
    setError(null)
  }

  async function handleAnimate() {
    if (!file || !animationPrompt.trim()) return
    setLoading(true)
    setError(null)
    try {
      const result = await uploadImage(file)
      setUploadResult(result)
      setStage('confirm')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleEdit() {
    setStage('upload')
    setUploadResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  async function handleStartAnimation() {
    setLoading(true)
    setError(null)
    try {
      const result = await submitAnimation(uploadResult.image_id, animationPrompt)
      if (onAnimateSuccess) {
        onAnimateSuccess(result, {
          animation_prompt: animationPrompt,
          image_id: uploadResult.image_id,
        })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (stage === 'confirm' && uploadResult) {
    return (
      <div className="form-wrapper">
        <h2 className="result-title">Ready to animate</h2>
        <div className="result-panel">
          <img
            src={`${API_BASE}${uploadResult.preview_url}`}
            alt="Uploaded image preview"
            style={{ width: '100%', maxWidth: 300, display: 'block', borderRadius: 4, marginBottom: '0.75rem' }}
          />
          <div className="result-row">
            <span className="result-label">Dimensions</span>
            <span className="result-value">{uploadResult.width} × {uploadResult.height}</span>
          </div>
          <div className="result-row">
            <span className="result-label">Animation</span>
            <span className="result-value">{animationPrompt}</span>
          </div>
        </div>
        {error && (
          <div className="error-message" role="alert">
            {error}
          </div>
        )}
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
          <button
            type="button"
            className="submit-btn"
            style={{ flex: '0 0 auto', width: 'auto', padding: '0.75rem 1.25rem' }}
            onClick={handleEdit}
          >
            ← Edit
          </button>
          <button
            type="button"
            className="submit-btn"
            style={{ flex: 1 }}
            disabled={loading}
            onClick={handleStartAnimation}
          >
            {loading ? 'Submitting…' : 'Start Animation'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="form-wrapper">
      <div className="field-set">
        <legend style={{ fontWeight: 700, fontSize: '0.95rem', display: 'block', marginBottom: '0.25rem' }}>
          Upload your background image
        </legend>
        <p className="field-driver">
          JPEG, PNG or WebP. Best results with 16:9 images. Avoid faces and text.
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <button
          type="button"
          className="submit-btn"
          style={{ width: 'auto', padding: '0.5rem 1rem', marginTop: 0 }}
          onClick={() => fileInputRef.current?.click()}
        >
          Choose Image
        </button>
        {localPreviewUrl && (
          <img
            src={localPreviewUrl}
            alt="Local image preview"
            style={{ display: 'block', maxWidth: 300, marginTop: '0.75rem', borderRadius: 4 }}
          />
        )}
      </div>
      <div className="field-set">
        <legend style={{ fontWeight: 700, fontSize: '0.95rem', display: 'block', marginBottom: '0.25rem' }}>
          Describe what to animate
        </legend>
        <p className="field-driver">
          e.g. Light streaks sliding left to right, world map glowing, subtle shimmer on panels
        </p>
        <textarea
          rows={4}
          style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem', border: '1px solid #d0d5dd', borderRadius: '4px', background: '#fafafa', color: '#1a1a1a', resize: 'vertical', fontFamily: 'inherit' }}
          value={animationPrompt}
          onChange={(e) => setAnimationPrompt(e.target.value)}
        />
      </div>
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      <button
        type="button"
        className="submit-btn"
        disabled={!file || !animationPrompt.trim() || loading}
        onClick={handleAnimate}
      >
        {loading ? 'Uploading…' : 'Animate →'}
      </button>
    </div>
  )
}
