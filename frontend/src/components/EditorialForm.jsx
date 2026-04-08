import React, { useState } from 'react'
import FORM_OPTIONS from '../config/formOptions.js'
import { compilePrompts } from '../api/client.js'

const FIELD_META = {
  category:          { label: 'Category',          driver: 'Drives positive prompt fragment' },
  location_feel:     { label: 'Location Feel',      driver: 'Drives location prompt fragment' },
  time_of_day:       { label: 'Time of Day',        driver: 'Drives time-of-day prompt fragment' },
  color_temperature: { label: 'Color Temperature',  driver: 'Drives LUT grade selection' },
  mood:              { label: 'Mood',               driver: 'Drives mood prompt fragment' },
  motion_intensity:  { label: 'Motion Intensity',   driver: 'Drives motion prompt fragment' },
}

const INITIAL_FORM = {
  category: '',
  location_feel: '',
  time_of_day: '',
  color_temperature: '',
  mood: '',
  motion_intensity: '',
}

export default function EditorialForm({ onCompileSuccess }) {
  const [formData, setFormData] = useState(INITIAL_FORM)
  const [compileResult, setCompileResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const allFieldsFilled = Object.values(formData).every((v) => v !== '')

  function handleChange(e) {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()

    if (!Object.values(formData).every((v) => v !== '')) {
      return
    }

    setIsLoading(true)
    setError(null)
    setCompileResult(null)

    try {
      const result = await compilePrompts(formData)
      setCompileResult(result)
      if (onCompileSuccess) {
        onCompileSuccess(result, formData)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="form-wrapper">
      <form onSubmit={handleSubmit} noValidate>
        {Object.entries(FIELD_META).map(([fieldKey, meta]) => (
          <fieldset key={fieldKey} className="field-set">
            <legend>{meta.label}</legend>
            <p className="field-driver">{meta.driver}</p>
            <select
              name={fieldKey}
              value={formData[fieldKey]}
              onChange={handleChange}
              className="field-select"
            >
              <option value="" disabled>
                Select {meta.label}...
              </option>
              {FORM_OPTIONS[fieldKey].map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} — {opt.helper}
                </option>
              ))}
            </select>
          </fieldset>
        ))}

        {error && (
          <div className="error-message" role="alert">
            {error}
          </div>
        )}

        <button
          type="submit"
          className="submit-btn"
          disabled={!allFieldsFilled || isLoading}
        >
          {isLoading ? 'Compiling...' : 'Compile Prompts'}
        </button>
      </form>

      {compileResult && (
        <div className="result-panel">
          <h2 className="result-title">Compile Result</h2>
          <div className="result-row">
            <span className="result-label">Run ID</span>
            <span className="result-value">{compileResult.input_hash_short}</span>
          </div>
          <div className="result-row">
            <span className="result-label">LUT Grade</span>
            <span className="result-value">{compileResult.selected_lut}</span>
          </div>
          <div className="result-row">
            <span className="result-label">Lower Third Style</span>
            <span className="result-value">{compileResult.lower_third_style}</span>
          </div>
          <div className="result-row">
            <span className="result-label">Prompt Hash</span>
            <span className="result-value">{compileResult.positive_hash.slice(0, 8)}</span>
          </div>
          <button className="proceed-btn" disabled>
            Proceed to Generation → (available in next step)
          </button>
        </div>
      )}
    </div>
  )
}
