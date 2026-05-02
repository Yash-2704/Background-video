import React, { useState, useEffect, useRef } from 'react'
import { submitGeneration, getRunStatus, getMediaUrl } from '../api/client.js'

// 11 stages — no interpolation stage
// (Wan2.2-TI2V-5B outputs 24fps natively)
const PIPELINE_STAGES = [
  { key: 'prompt_compilation', label: 'Prompt Compilation' },
  { key: 'generation', label: 'Video Generation' },
  { key: 'probe_decode', label: 'Decode Probe' },
  { key: 'probe_temporal', label: 'Temporal Probe', skipped: true },
  { key: 'gate_evaluation', label: 'Quality Gates', skipped: true },
  { key: 'upscale', label: 'Upscale 1080p', skipped: true },
  { key: 'mask_generation', label: 'Mask Generation', skipped: true },
  { key: 'lut_grading', label: 'LUT Grading', skipped: true },
  { key: 'composite', label: 'Composite', skipped: true },
  { key: 'preview_export', label: 'Preview Export', skipped: true },
  { key: 'metadata_assembly', label: 'Metadata Assembly', skipped: true },
]

const FIELD_LABELS = {
  category: 'Category',
  location_feel: 'Location Feel',
  time_of_day: 'Time of Day',
  color_temperature: 'Color Temperature',
  mood: 'Mood',
  motion_intensity: 'Motion Intensity',
}

const TERMINAL_STATUSES = new Set(['complete', 'failed', 'escalated'])

const INITIAL_STAGE_STATUSES = Object.fromEntries(
  PIPELINE_STAGES.map((s) => [s.key, 'idle'])
)

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function parseStages(stagesData) {
  if (!stagesData) return {}
  if (Array.isArray(stagesData)) {
    return Object.fromEntries(stagesData.map((s) => [s.stage, s.status]))
  }
  return stagesData
}

function StageIndicator({ status }) {
  if (status === 'complete') return <span className="stage-indicator stage-complete">✓</span>
  if (status === 'failed') return <span className="stage-indicator stage-failed">✕</span>
  if (status === 'running') return <span className="stage-indicator stage-running" />
  return <span className="stage-indicator stage-idle" />
}

export default function RunMonitor({ compileResult, formData, onBack, onComplete }) {
  const [runId, setRunId] = useState(null)
  const [runStatus, setRunStatus] = useState('starting')
  const [stageStatuses, setStageStatuses] = useState(INITIAL_STAGE_STATUSES)
  const [runResult, setRunResult] = useState(null)
  const [error, setError] = useState(null)
  const [elapsedMs, setElapsedMs] = useState(0)

  // Refs to allow cleanup from async callbacks without stale closures
  const pollRef = useRef(null)
  const elapsedRef = useRef(null)
  const cancelledRef = useRef(false)

  useEffect(() => {
    cancelledRef.current = false

    elapsedRef.current = setInterval(() => {
      setElapsedMs((prev) => prev + 1000)
    }, 1000)

    function stopIntervals() {
      clearInterval(pollRef.current)
      clearInterval(elapsedRef.current)
    }

    async function startRun() {
      setError(null)
      try {
        const response = await submitGeneration({
          run_id: compileResult.input_hash_short,
          compiled: compileResult,
        })

        if (cancelledRef.current) return

        const id = response.run_id
        setRunId(id)
        setRunStatus('running')

        pollRef.current = setInterval(async () => {
          if (cancelledRef.current) return
          try {
            const statusResponse = await getRunStatus(id)
            if (cancelledRef.current) return

            if (statusResponse.stages && Object.keys(statusResponse.stages).length > 0) {
              setStageStatuses((prev) => ({
                ...prev,
                ...parseStages(statusResponse.stages),
              }))
            }

            if (TERMINAL_STATUSES.has(statusResponse.status)) {
              stopIntervals()
              setRunResult(statusResponse)
              setRunStatus(statusResponse.status)
              if (statusResponse.status === 'complete' && statusResponse.gate_result?.overall !== 'raw_verify') {
                onComplete(statusResponse)
              }
            }
          } catch (err) {
            if (!cancelledRef.current) {
              stopIntervals()
              setError(err.message)
            }
          }
        }, 60000)
      } catch (err) {
        if (!cancelledRef.current) {
          stopIntervals()
          setError(err.message)
          setRunStatus('failed')
        }
      }
    }

    startRun()

    return () => {
      cancelledRef.current = true
      clearInterval(pollRef.current)
      clearInterval(elapsedRef.current)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const isTerminal = TERMINAL_STATUSES.has(runStatus)

  function statusBannerClass() {
    if (runStatus === 'complete') return 'status-banner status-complete'
    if (runStatus === 'failed') return 'status-banner status-failed'
    if (runStatus === 'escalated') return 'status-banner status-escalated'
    return 'status-banner status-running'
  }

  function statusBannerText() {
    if (runStatus === 'starting') return 'Initializing run...'
    if (runStatus === 'running') return 'Pipeline running'
    if (runStatus === 'complete') return 'Run complete'
    if (runStatus === 'failed') return 'Run failed'
    if (runStatus === 'escalated') return 'Escalated to human review'
    return ''
  }

  return (
    <div className="monitor-wrapper">
      {/* Header */}
      <div className="monitor-header">
        <span className="monitor-run-id">Run: {compileResult.input_hash_short}</span>
        <span className="monitor-elapsed">{formatElapsed(elapsedMs)}</span>
        <button className="back-btn" onClick={() => onBack()}>← New Run</button>
      </div>

      {/* Run summary strip */}
      <div className="run-summary-strip">
        {compileResult?.mode === 'i2v' ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0' }}>
            <img
              src={getMediaUrl(compileResult.image_id, `${compileResult.image_id}.jpg`)}
              alt="Source"
              style={{ height: 60, borderRadius: 4, objectFit: 'cover' }}
              onError={(e) => { e.target.style.display = 'none' }}
            />
            <div>
              <span className="summary-label">Animation:</span>
              <span className="summary-value" style={{ marginLeft: 6 }}>
                {formData?.animation_prompt ?? '—'}
              </span>
            </div>
          </div>
        ) : (
          Object.entries(FIELD_LABELS).map(([key, label]) => (
            <div key={key} className="summary-item">
              <span className="summary-label">{label}:</span>
              <span className="summary-value">{formData?.[key] ?? '—'}</span>
            </div>
          ))
        )}
      </div>

      {/* Overall status banner */}
      <div className={statusBannerClass()} role="status">
        {statusBannerText()}
      </div>

      {/* Error display */}
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      {/* Stage progress list */}
      <ul className="stage-list">
        {PIPELINE_STAGES.map((stage) => {
          const status = stageStatuses[stage.key] || 'idle'
          return (
            <li
              key={stage.key}
              className={`stage-row stage-row--${status}`}
              data-testid={`stage-${stage.key}`}
              data-status={status}
            >
              <StageIndicator status={status} />
              <span className={`stage-label${status === 'failed' ? ' stage-label--failed' : ''}`}>
                {stage.label}{stage.skipped ? <span style={{ color: '#888', fontSize: '0.8em', marginLeft: 6 }}>Skipped</span> : null}
              </span>
            </li>
          )
        })}
      </ul>

      {/* Result panel — visible only when complete */}
      {runStatus === 'complete' && runResult && (
        <div className="result-panel">
          <h2 className="result-title">Run Result</h2>
          <div className="result-row">
            <span className="result-label">Raw Loop</span>
            <span className="result-value">{runResult.raw_loop_path}</span>
          </div>
          <div className="result-row">
            <span className="result-label">Seed</span>
            <span className="result-value">{runResult.seed}</span>
          </div>
          <div className="result-row">
            <span className="result-label">Seam Frames</span>
            <span className="result-value">
              {Array.isArray(runResult.seam_frames_playable)
                ? runResult.seam_frames_playable.join(', ')
                : runResult.seam_frames_playable}
            </span>
          </div>
          <div className="result-row">
            <span className="result-label">Gate Result</span>
            <span className="result-value">{runResult.gate_result?.overall}</span>
          </div>
          {runResult.gate_result?.overall === 'raw_verify' ? (
            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ marginBottom: '0.5rem' }}>Raw Output Preview</h3>
              <video
                controls
                loop
                src={getMediaUrl(
                  runResult.run_id,
                  runResult.upscaled_loop_path
                    ? `${runResult.run_id}_1080p.mp4`
                    : `bg_${runResult.run_id}_raw_loop.mp4`
                )}
                style={{ width: '100%', maxWidth: '720px', display: 'block' }}
              />
              <div className="result-row" style={{ marginTop: '0.5rem' }}>
                <span className="result-label">Seed</span>
                <span className="result-value">{runResult.seed}</span>
              </div>
              <p style={{ marginTop: '0.5rem', fontStyle: 'italic', fontSize: '0.85em' }}>
                Raw unprocessed output — no upscale, grade, or composite applied.
              </p>
            </div>
          ) : (
            <button
              className="output-bundle-btn"
              onClick={() => onComplete(runResult)}
            >
              View Output Bundle →
            </button>
          )}
        </div>
      )}
    </div>
  )
}
