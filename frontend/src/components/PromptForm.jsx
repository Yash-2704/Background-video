import React, { useState } from 'react'
import { compilePrompts } from '../api/client.js'

export default function PromptForm({ onCompileSuccess }) {
  const [prompt, setPrompt]   = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  async function handleGenerate() {
    setLoading(true)
    setError(null)
    try {
      const result = await compilePrompts(prompt)
      if (onCompileSuccess) {
        onCompileSuccess(result, { raw_prompt: prompt })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
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
        onClick={handleGenerate}
      >
        {loading ? 'Generating…' : 'Generate →'}
      </button>
    </div>
  )
}
