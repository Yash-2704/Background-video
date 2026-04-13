import React, { useState, useEffect, useCallback } from 'react'
import { fetchBundleFile } from '../api/client.js'

const _MEDIA_BASE = 'http://localhost:8000/api/v1/media'
function _mediaUrl(clip_id, filename) {
  return `${_MEDIA_BASE}/${clip_id}/${filename}`
}

// ── Tab definitions ────────────────────────────────────────────────────────────

const VIEWER_TABS = [
  {
    key:      'metadata',
    label:    'Generation Record',
    filename: (clip_id) => `${clip_id}_metadata.json`,
    audience: 'What happened',
  },
  {
    key:      'edit_manifest',
    label:    'Edit Manifest',
    filename: (clip_id) => `${clip_id}_edit_manifest.json`,
    audience: 'What can change',
  },
  {
    key:      'contract',
    label:    'Integration Contract',
    filename: (clip_id) => `${clip_id}_integration_contract.json`,
    audience: 'How to consume this asset',
  },
  {
    key:      'generation_log',
    label:    'Generation Log',
    filename: (clip_id) => `${clip_id}_generation_log.json`,
    audience: 'Run history',
  },
]

// ── InfoRow sub-component ─────────────────────────────────────────────────────

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span className="info-label">{label}</span>
      <span className="info-value">{String(value ?? '—')}</span>
    </div>
  )
}

// ── Section header ─────────────────────────────────────────────────────────────

function SectionHeader({ title }) {
  return (
    <div className="section-header">
      <h3 className="section-title">{title}</h3>
      <hr className="section-rule" />
    </div>
  )
}

// ── Gate / integrity badge ─────────────────────────────────────────────────────

function GateBadge({ value }) {
  const cls =
    value === 'pass'
      ? 'gate-badge gate-pass'
      : value === 'fail'
      ? 'gate-badge gate-fail'
      : 'gate-badge gate-review'
  return (
    <span className={cls} data-gate={value}>
      {value}
    </span>
  )
}

// ── Cost badge ────────────────────────────────────────────────────────────────

function CostBadge({ cost }) {
  const cls =
    cost === 'none'
      ? 'cost-badge cost-none'
      : cost === 'extension_pass'
      ? 'cost-badge cost-amber'
      : 'cost-badge cost-blue'
  return <span className={cls}>{cost}</span>
}

// ── Tab 1: Generation Record ───────────────────────────────────────────────────

function MetadataPanel({ data }) {
  const g  = data.generation         || {}
  const pp = data.prompt_provenance  || {}
  const ui = data.user_input         || {}
  const qg = data.quality_gates      || {}
  const po = data.post_processing    || {}

  const fpsDisplay = `${g.native_fps ?? '?'}fps (native)`
  const positiveHash = pp.positive_hash
    ? `${pp.positive_hash.slice(0, 8)}...`
    : '—'

  const files = data.files || {}
  const finalPath   = files.final   && files.final   !== 'MISSING' ? files.final   : null
  const previewPath = files.preview && files.preview !== 'MISSING' ? files.preview : null
  const videoFilename = finalPath   ? finalPath.split('/').pop()   : null
  const gifFilename   = previewPath ? previewPath.split('/').pop() : null
  const videoUrl = videoFilename ? _mediaUrl(data.clip_id, videoFilename) : null
  const gifUrl   = gifFilename   ? _mediaUrl(data.clip_id, gifFilename)   : null

  return (
    <div className="tab-panel">

      {videoUrl && (
        <section className="video-preview-section">
          <h3 className="video-preview-title">Final Composite</h3>
          <video
            className="video-player"
            src={videoUrl}
            controls
            loop
            playsInline
            data-testid="final-video-player"
          />
          <p className="video-preview-label">
            {data.clip_id} · {g.playable_duration_s}s · {g.native_fps}fps · loop
          </p>
        </section>
      )}

      {gifUrl && (
        <section className="gif-preview-section">
          <h3 className="video-preview-title">Preview GIF (3-segment)</h3>
          <img
            className="gif-player"
            src={gifUrl}
            alt="Three-segment preview GIF"
            data-testid="preview-gif"
          />
        </section>
      )}

      <SectionHeader title="Run Info" />
      <InfoRow label="Clip ID"          value={data.clip_id} />
      <InfoRow label="Module Version"   value={data.module_version} />
      <InfoRow label="Schema Version"   value={data.schema_version} />
      <InfoRow label="Timestamp (UTC)"  value={data.timestamp_utc} />
      <InfoRow label="Dev Mode"         value={String(data.dev_mode)} />

      <SectionHeader title="Generation" />
      <InfoRow label="Model"                    value={g.model} />
      <InfoRow label="Sampler"                  value={g.sampler} />
      <InfoRow label="CFG Scale"                value={g.cfg_scale} />
      <InfoRow label="Steps"                    value={g.steps} />
      <InfoRow label="Seed"                     value={g.seed} />
      <InfoRow label="Frame Rate"               value={fpsDisplay} />
      <InfoRow label="Base Duration (s)"        value={g.base_duration_s} />
      <InfoRow label="Extensions"               value={g.extensions} />
      <InfoRow label="Total Raw Duration (s)"   value={g.total_raw_duration_s} />
      <InfoRow label="Playable Duration (s)"    value={g.playable_duration_s} />
      <InfoRow label="Crossfade Frames"         value={g.crossfade_frames} />
      <InfoRow
        label="Seam Frames (Playable)"
        value={Array.isArray(g.seam_frames_playable) ? g.seam_frames_playable.join(', ') : g.seam_frames_playable}
      />

      <SectionHeader title="Prompt Provenance" />
      <InfoRow label="Compiler Version"         value={pp.compiler_version} />
      <InfoRow label="Input Hash"               value={pp.input_hash_short} />
      <InfoRow label="Positive Hash"            value={positiveHash} />
      <InfoRow label="Reproducibility Rating"   value={pp.reproducibility_rating} />

      <SectionHeader title="User Input" />
      <InfoRow label="Category"          value={ui.category} />
      <InfoRow label="Location Feel"     value={ui.location_feel} />
      <InfoRow label="Time of Day"       value={ui.time_of_day} />
      <InfoRow label="Color Temperature" value={ui.color_temperature} />
      <InfoRow label="Mood"              value={ui.mood} />
      <InfoRow label="Motion Intensity"  value={ui.motion_intensity} />

      <SectionHeader title="Quality Gates" />
      <InfoRow label="Mean Luminance"         value={qg.mean_luminance} />
      <InfoRow label="Luminance Gate"         value={qg.luminance_gate} />
      <InfoRow label="Flicker Index"          value={qg.flicker_index} />
      <InfoRow label="Warping Artifact Score" value={qg.warping_artifact_score} />
      <InfoRow label="Scene Cut Detected"     value={String(qg.scene_cut_detected)} />
      <InfoRow label="Perceptual Loop Score"  value={qg.perceptual_loop_score} />
      <div className="info-row">
        <span className="info-label">Overall</span>
        <span className="info-value"><GateBadge value={qg.overall} /></span>
      </div>

      <SectionHeader title="Post Processing" />
      <InfoRow label="Anchor Position"  value={po.anchor_position_default} />
      <InfoRow label="Selected LUT"     value={po.selected_lut} />
      <InfoRow
        label="LUTs Generated"
        value={Array.isArray(po.luts_generated) ? po.luts_generated.join(', ') : po.luts_generated}
      />
      <InfoRow
        label="Masks Generated"
        value={Array.isArray(po.masks_generated) ? po.masks_generated.join(', ') : po.masks_generated}
      />

      <SectionHeader title="Files" />
      {(() => {
        const f = data.files || {}
        return (
          <>
            <InfoRow label="Raw Loop"       value={f.raw_loop} />
            <InfoRow label="Upscaled"       value={f.upscaled} />
            <InfoRow label="Decode Probe"   value={f.decode_probe} />
            <InfoRow label="Temporal Probe" value={f.temporal_probe} />
            <InfoRow label="Final"          value={f.final} />
            <InfoRow label="Preview"        value={f.preview} />
          </>
        )
      })()}
    </div>
  )
}

// ── Tab 2: Edit Manifest ───────────────────────────────────────────────────────

const PARAM_LABELS = {
  anchor_zone:              'Anchor Zone',
  luminance_reduction:      'Luminance Reduction',
  lut:                      'LUT',
  clip_start_offset_s:      'Clip Start Offset (s)',
  loop_duration_extension:  'Loop Duration Extension',
}

function EditManifestPanel({ data }) {
  const params  = data.editable_parameters || {}
  const locked  = data.locked_parameters   || {}

  return (
    <div className="tab-panel">

      <SectionHeader title="Source Files" />
      <InfoRow label="Source Raw"      value={data.source_raw} />
      <InfoRow label="Source Upscaled" value={data.source_upscaled} />

      <SectionHeader title="Editable Parameters" />
      {Object.entries(params).map(([key, param]) => (
        <div key={key} className="param-card">
          <div className="param-name">{PARAM_LABELS[key] || key}</div>
          <InfoRow label="Current Value"  value={param.current ?? param.current_s} />
          <div className="info-row">
            <span className="info-label">Cost</span>
            <span className="info-value"><CostBadge cost={param.re_render_cost} /></span>
          </div>
          <InfoRow label="Est. Time" value={`${param.estimated_time_s}s`} />
          <InfoRow label="Operation" value={param.operation} />
          {param.warning && (
            <div className="param-warning">{param.warning}</div>
          )}
        </div>
      ))}

      <SectionHeader title="Locked Parameters" />
      <div className="locked-warning">
        <InfoRow label="Model" value={locked.model} />
        <InfoRow label="Seed"  value={locked.seed} />
        <div className="locked-note">{locked.note}</div>
      </div>
    </div>
  )
}

// ── Tab 3: Integration Contract ────────────────────────────────────────────────

function ContractPanel({ data }) {
  const he  = data.for_human_editor        || {}
  const dm  = data.for_downstream_modules  || {}
  const cp  = data.for_compositor_process  || {}
  const sd  = dm.scene_descriptors         || {}
  const ms  = dm.module_suggestions        || {}
  const res = Array.isArray(he.resolution)
    ? `${he.resolution[0]} × ${he.resolution[1]}`
    : he.resolution

  return (
    <div className="tab-panel">

      <SectionHeader title="For Human Editor" />
      <InfoRow label="Primary Asset"    value={he.primary_asset} />
      <InfoRow label="Duration (s)"     value={he.duration_s} />
      <InfoRow label="FPS"              value={he.fps} />
      <InfoRow label="Resolution"       value={res} />
      <InfoRow label="Is Loop"          value={String(he.is_loop)} />
      <InfoRow label="Loop Clean"       value={String(he.loop_clean)} />
      <InfoRow
        label="Recommended Cut Points (s)"
        value={Array.isArray(he.recommended_cut_points_s)
          ? he.recommended_cut_points_s.join(', ')
          : he.recommended_cut_points_s}
      />
      {he.editor_notes && (
        <div className="editor-notes-callout">{he.editor_notes}</div>
      )}

      <SectionHeader title="For Downstream Modules" />
      <InfoRow label="Dominant Environment"   value={sd.dominant_environment} />
      <InfoRow label="Light Condition"         value={sd.light_condition} />
      <InfoRow label="Motion Character"        value={sd.motion_character} />
      <InfoRow label="Color Temperature (K)"   value={sd.color_temperature_k} />
      <InfoRow label="Background Complexity"   value={sd.background_complexity} />
      <InfoRow label="Lower Third Style"       value={ms.lower_third_style} />
      <InfoRow label="Ticker Contrast"         value={ms.ticker_contrast} />
      <InfoRow label="Audio Mood Match"        value={ms.audio_mood_match} />

      <SectionHeader title="For Compositor Process" />
      <InfoRow label="Primary Asset"           value={cp.primary_asset} />
      <InfoRow label="Frame Count"             value={cp.frame_count} />
      <InfoRow label="Verified Loop"           value={String(cp.verified_loop)} />
      <InfoRow label="Flicker Index"           value={cp.flicker_index} />
      <InfoRow label="Warping Artifact Score"  value={cp.warping_artifact_score} />
      <div className="info-row">
        <span className="info-label">Integrity Check</span>
        <span className="info-value"><GateBadge value={cp.integrity_check} /></span>
      </div>
      <InfoRow
        label="Loop Frame Delta"
        value={cp.loop_frame_delta_max?.value}
      />
    </div>
  )
}

// ── Tab 4: Generation Log ──────────────────────────────────────────────────────

function GenerationLogPanel({ data }) {
  const failureLog = Array.isArray(data.failure_log) ? data.failure_log : []

  return (
    <div className="tab-panel">

      <SectionHeader title="Run Summary" />
      <InfoRow label="Clip ID"             value={data.clip_id} />
      <InfoRow label="Total Attempts"      value={data.total_attempts} />
      <InfoRow label="Outcome"             value={data.outcome} />
      <InfoRow label="Escalated to Human"  value={data.escalated_to_human ? 'Yes' : 'No'} />
      <InfoRow
        label="Seeds Used"
        value={Array.isArray(data.seeds_used) ? data.seeds_used.join(', ') : data.seeds_used}
      />

      <SectionHeader title="Attempt History" />
      {failureLog.length === 0 ? (
        <p className="no-retry-message">Run passed on first attempt. No retry history.</p>
      ) : (
        failureLog.map((attempt, i) => (
          <div key={i} className="attempt-card">
            <div className="attempt-number">Attempt {attempt.attempt ?? i + 1}</div>
            <InfoRow label="Seed"        value={attempt.seed} />
            <InfoRow label="CFG Used"    value={attempt.cfg_used} />
            <InfoRow label="Gate Result" value={attempt.gate_result} />
            {attempt.failures && attempt.failures.length > 0 && (
              <div className="attempt-failures">
                <span className="attempt-failures-label">Failures:</span>
                <ul>
                  {attempt.failures.map((f, j) => <li key={j}>{f}</li>)}
                </ul>
              </div>
            )}
            {attempt.human_flags && attempt.human_flags.length > 0 && (
              <div className="attempt-failures">
                <span className="attempt-failures-label">Human Flags:</span>
                <ul>
                  {attempt.human_flags.map((f, j) => <li key={j}>{f}</li>)}
                </ul>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  )
}

// ── Main BundleViewer component ───────────────────────────────────────────────

export default function BundleViewer({ runResult, onBack }) {
  const clip_id = runResult?.run_id

  const [activeTab, setActiveTab] = useState('metadata')
  const [tabData,   setTabData]   = useState({})
  const [tabError,  setTabError]  = useState({})

  const fetchTab = useCallback(async (key) => {
    const tab = VIEWER_TABS.find((t) => t.key === key)
    if (!tab) return

    setTabData((prev) => ({ ...prev, [key]: 'loading' }))
    setTabError((prev) => ({ ...prev, [key]: null }))

    try {
      const filename = tab.filename(clip_id)
      const result   = await fetchBundleFile(clip_id, filename)
      if (result === null) {
        setTabData((prev) => ({ ...prev, [key]: null }))
      } else {
        setTabData((prev) => ({ ...prev, [key]: result }))
      }
    } catch (err) {
      setTabData((prev) => ({ ...prev, [key]: 'error' }))
      setTabError((prev) => ({ ...prev, [key]: err.message }))
    }
  }, [clip_id])

  // Fetch metadata tab on mount
  useEffect(() => {
    fetchTab('metadata')
  }, [fetchTab])

  function handleTabClick(key) {
    setActiveTab(key)
    if (tabData[key] === undefined) {
      fetchTab(key)
    }
  }

  function renderTabContent(key) {
    const data = tabData[key]

    if (data === undefined || data === 'loading') {
      return <p className="tab-loading">Loading...</p>
    }
    if (data === null || data === 'error') {
      return <p className="tab-unavailable">File not available for this run.</p>
    }

    if (key === 'metadata')       return <MetadataPanel      data={data} />
    if (key === 'edit_manifest')  return <EditManifestPanel  data={data} />
    if (key === 'contract')       return <ContractPanel      data={data} />
    if (key === 'generation_log') return <GenerationLogPanel data={data} />
    return null
  }

  return (
    <div className="viewer-wrapper">

      {/* Header */}
      <div className="viewer-header">
        <span className="viewer-clip-id">Output Bundle — {clip_id}</span>
        <button className="back-btn viewer-back-btn" onClick={onBack}>
          ← Back to Monitor
        </button>
      </div>

      {/* Tab bar */}
      <div className="viewer-tab-bar" role="tablist">
        {VIEWER_TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            className={`tab-btn${activeTab === tab.key ? ' tab-btn--active' : ''}`}
            onClick={() => handleTabClick(tab.key)}
          >
            <span className="tab-label">{tab.label}</span>
            <span className="tab-audience">{tab.audience}</span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="viewer-content" role="tabpanel">
        {renderTabContent(activeTab)}
      </div>
    </div>
  )
}
