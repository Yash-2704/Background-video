import React, { useState } from 'react'

const INITIAL_FORM = {
  category: 'Breaking News',
  location_feel: 'Urban Exterior',
  time_of_day: 'Morning',
  color_temperature: 'Neutral',
  mood: 'Calm',
  motion_intensity: 0.5,
}

export default function PrototypeViewer() {
  const [formData, setFormData] = useState(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  function handleChange(e) {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'motion_intensity' ? parseFloat(value) : value,
    }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/prototype/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const body = await response.json()
        setError(body.detail ?? 'Unknown error')
        setLoading(false)
        return
      }

      const data = await response.json()
      setResult(data)
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '700px', margin: '2rem auto', fontFamily: 'sans-serif' }}>
      <h2>Prototype Generator</h2>

      <form onSubmit={handleSubmit}>
        {/* category */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Category
          </label>
          <select name="category" value={formData.category} onChange={handleChange}>
            {['Breaking News', 'Politics', 'Business', 'Sports', 'Weather', 'Entertainment', 'Technology', 'Health'].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* location_feel */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Location Feel
          </label>
          <select name="location_feel" value={formData.location_feel} onChange={handleChange}>
            {['Indoor Studio', 'Urban Exterior', 'Natural Landscape', 'Abstract'].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* time_of_day */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Time of Day
          </label>
          <select name="time_of_day" value={formData.time_of_day} onChange={handleChange}>
            {['Morning', 'Midday', 'Evening', 'Night'].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* color_temperature */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Color Temperature
          </label>
          <select name="color_temperature" value={formData.color_temperature} onChange={handleChange}>
            {['Warm', 'Neutral', 'Cool', 'Desaturated'].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* mood */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Mood
          </label>
          <select name="mood" value={formData.mood} onChange={handleChange}>
            {['Urgent', 'Calm', 'Dramatic', 'Optimistic', 'Serious'].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* motion_intensity */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.25rem' }}>
            Motion Intensity: {formData.motion_intensity.toFixed(2)}
          </label>
          <input
            type="range"
            name="motion_intensity"
            min="0"
            max="1"
            step="0.01"
            value={formData.motion_intensity}
            onChange={handleChange}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '0.6rem 1.6rem',
            fontSize: '1rem',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Generating…' : 'Generate'}
        </button>
      </form>

      {loading && (
        <p>Generating… (~35 seconds)</p>
      )}

      {error && (
        <div
          style={{
            color: 'red',
            marginTop: '1rem',
            padding: '0.75rem',
            border: '1px solid red',
            borderRadius: '4px',
          }}
        >
          {error}
        </div>
      )}

      {result && (
        <div>
          <video
            src={'/' + result.video_path}
            autoPlay
            loop
            muted
            playsInline
            controls
            width="640"
          />
          <img
            src={'/' + result.image_path}
            alt="Generated still"
            width="640"
            style={{ display: 'block', marginTop: '1rem' }}
          />
          <p><strong>Run ID:</strong> {result.run_id}</p>
          <p><strong>Prompt used:</strong></p>
          <pre style={{ whiteSpace: 'pre-wrap', background: '#f4f4f4', padding: '0.75rem' }}>
            {result.prompt_used}
          </pre>
        </div>
      )}
    </div>
  )
}
