import React, { useState } from 'react'
import { parsePrompt, compilePrompts } from '../api/client.js'

const FIELD_LABELS = {
  category:          'Category',
  location_feel:     'Location Feel',
  time_of_day:       'Time of Day',
  color_temperature: 'Color Temperature',
  mood:              'Mood',
  motion_intensity:  'Motion Intensity',
}

export default function PromptForm({ onCompileSuccess }) {
  const [prompt, setPrompt]   = useState('')
  const [stage, setStage]     = useState('input')
  const [parsed, setParsed]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  async function handleInterpret() {
    setLoading(true)
    setError(null)
    try {
      const result = await parsePrompt(prompt)
      setParsed(result)
      setStage('confirm')
    } catch (err) {
      setError(
        err.message.includes('500')
          ? 'The interpret service is not available yet. Please contact the operator.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  function handleEdit() {
    setStage('input')
    setError(null)
  }

  async function handleConfirm() {
    setLoading(true)
    setError(null)
    const inferredFields = {
      category:          parsed.category,
      location_feel:     parsed.location_feel,
      time_of_day:       parsed.time_of_day,
      color_temperature: parsed.color_temperature,
      mood:              parsed.mood,
      motion_intensity:  parsed.motion_intensity,
    }
    try {
      const result = await compilePrompts(inferredFields)
      if (onCompileSuccess) {
        onCompileSuccess(result, inferredFields)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (stage === 'confirm' && parsed) {
    return (
      <div className="form-wrapper">
        <h2 className="result-title">Interpreted as:</h2>
        <div className="result-panel">
          {Object.entries(FIELD_LABELS).map(([key, label]) => (
            <div key={key} className="result-row">
              <span className="result-label">{label}</span>
              <span className="result-value">{parsed[key]}</span>
            </div>
          ))}
        </div>
        {parsed.inference_notes && (
          <p style={{ fontSize: '0.82rem', fontStyle: 'italic', color: '#6b7280', marginTop: '0.75rem' }}>
            {parsed.inference_notes}
          </p>
        )}
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
            onClick={handleConfirm}
          >
            {loading ? 'Generating…' : 'Confirm & Generate'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="form-wrapper">
      <div className="field-set">
        <legend style={{ fontWeight: 700, fontSize: '0.95rem', display: 'block', marginBottom: '0.25rem' }}>
          Describe the video you want
        </legend>
        <p className="field-driver">
          e.g. A tense night scene in a government plaza, cool tones, slow drift
        </p>
        <textarea
          rows={4}
          style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem', border: '1px solid #d0d5dd', borderRadius: '4px', background: '#fafafa', color: '#1a1a1a', resize: 'vertical', fontFamily: 'inherit' }}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
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
        disabled={!prompt.trim() || loading}
        onClick={handleInterpret}
      >
        {loading ? 'Interpreting…' : 'Interpret →'}
      </button>
    </div>
  )
}
