import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import BundleViewer from './BundleViewer.jsx'
import App from '../App.jsx'
import FORM_OPTIONS from '../config/formOptions.js'

// ── Mock API client ───────────────────────────────────────────────────────────

vi.mock('../api/client', () => ({
  submitGeneration: vi.fn(),
  getRunStatus:     vi.fn(),
  compilePrompts:   vi.fn(),
  fetchBundleFile:  vi.fn(),
}))

import {
  submitGeneration,
  getRunStatus,
  compilePrompts,
  fetchBundleFile,
} from '../api/client'

// ── Shared fixtures ───────────────────────────────────────────────────────────

const mockRunResult = {
  run_id:               'bg_001_b2e7f3',
  status:               'complete',
  raw_loop_path:        'raw/bg_001_b2e7f3_raw_loop.mp4',
  seed:                 42819,
  seam_frames_playable: [169, 338],
  gate_result:          { overall: 'pass' },
}

const mockMetadata = {
  clip_id:        'bg_001_b2e7f3',
  module_version: '1.1',
  schema_version: '1.0',
  timestamp_utc:  '2026-04-06T09:14:00Z',
  dev_mode:       true,
  generation: {
    model: 'CogVideoX-5b', sampler: 'DPM++ 2M',
    cfg_scale: 6.0, steps: 50, seed: 42819,
    native_fps: 8, interpolated_to_fps: 30,
    base_duration_s: 6, extensions: 2,
    total_raw_duration_s: 18, playable_duration_s: 17.067,
    crossfade_frames: 14,
    seam_frames_playable: [169, 338],
  },
  prompt_provenance: {
    compiler_version: '1.0.0',
    input_hash_short: 'b2e7f3',
    positive_hash:    'b2e7f3a1c9d4e8f2',
    reproducibility_rating: 'high',
  },
  user_input: {
    category: 'Economy', location_feel: 'Urban',
    time_of_day: 'Dusk', color_temperature: 'Cool',
    mood: 'Serious', motion_intensity: 'Gentle',
  },
  quality_gates: {
    mean_luminance: 0.46, luminance_gate: 'pass',
    flicker_index: 0.003, warping_artifact_score: 0.018,
    scene_cut_detected: false,
    perceptual_loop_score: 0.94, overall: 'pass',
  },
  post_processing: {
    anchor_position_default: 'center',
    selected_lut: 'cool_authority',
    luts_generated: ['cool_authority', 'neutral'],
    masks_generated: ['center', 'lower_third', 'upper_third'],
  },
  files: {
    raw_loop: 'raw/bg_001_b2e7f3_raw_loop.mp4',
    upscaled: 'raw/bg_001_b2e7f3_1080p.mp4',
    final:    'final/bg_001_b2e7f3.mp4',
    preview:  'final/bg_001_b2e7f3_preview.gif',
    masks:    [], luts: [],
  },
}

const mockEditManifest = {
  clip_id: 'bg_001_b2e7f3',
  edit_manifest_version: '1.0',
  source_raw:      'raw/bg_001_b2e7f3_raw_loop.mp4',
  source_upscaled: 'raw/bg_001_b2e7f3_1080p.mp4',
  editable_parameters: {
    anchor_zone: {
      current: 'center',
      re_render_cost: 'mask_only',
      estimated_time_s: 15,
      operation: 'swap mask file + recomposite',
      content_risk_by_position: {},
    },
    luminance_reduction: {
      current: 0.22, range: [0.10, 0.45],
      re_render_cost: 'mask_only',
      estimated_time_s: 15,
      operation: 'regenerate mask + recomposite',
    },
    lut: {
      current: 'cool_authority',
      available_precomputed: ['cool_authority', 'neutral'],
      re_render_cost: 'grade_only',
      estimated_time_s: 20,
      operation: 'apply new LUT to source_upscaled',
    },
    clip_start_offset_s: {
      current: 0, max: 12,
      re_render_cost: 'none',
      estimated_time_s: 2,
      operation: 'trim only',
    },
    loop_duration_extension: {
      current_s: 18, extendable_to_s: 30,
      re_render_cost: 'extension_pass',
      estimated_time_s: 900,
      operation: 'one additional diffusion extension',
      warning: 'Each extension pass introduces temporal drift risk.',
    },
  },
  locked_parameters: {
    model: 'CogVideoX-5b', seed: 42819,
    note: 'Any change requires full regeneration.',
  },
}

const mockContract = {
  clip_id: 'bg_001_b2e7f3',
  integration_contract_version: '1.0',
  for_human_editor: {
    primary_asset: 'final/bg_001_b2e7f3.mp4',
    duration_s: 17.067, fps: 30,
    resolution: [1920, 1080],
    is_loop: true, loop_clean: true,
    recommended_cut_points_s: [0, 5.63, 11.27, 17.07],
    editor_notes: 'Center anchor zone is pre-composited.',
  },
  for_downstream_modules: {
    scene_descriptors: {
      dominant_environment: 'urban_exterior',
      light_condition: 'dusk_directional',
      motion_character: 'slow_lateral_drift',
      color_temperature_k: 6500,
      background_complexity: 'medium',
    },
    module_suggestions: {
      lower_third_style: 'minimal_dark_bar',
      ticker_contrast: 'light_on_dark',
      audio_mood_match: 'measured_serious',
    },
  },
  for_compositor_process: {
    primary_asset: 'final/bg_001_b2e7f3.mp4',
    frame_count: 512,
    verified_loop: true,
    flicker_index: 0.003,
    warping_artifact_score: 0.018,
    integrity_check: 'pass',
    loop_frame_delta_max: { value: 0.003 },
  },
}

const mockGenerationLog = {
  clip_id: 'bg_001_b2e7f3',
  total_attempts: 1,
  seeds_used: [42819],
  outcome: 'pass_on_attempt_1',
  escalated_to_human: false,
  failure_log: [],
}

// ── Default props ─────────────────────────────────────────────────────────────

const defaultProps = {
  runResult: mockRunResult,
  onBack:    vi.fn(),
}

function renderViewer(props = {}) {
  return render(<BundleViewer {...defaultProps} {...props} />)
}

// ── Mock setup ────────────────────────────────────────────────────────────────

beforeEach(() => {
  vi.clearAllMocks()
  // Default: metadata tab resolves, others idle
  fetchBundleFile.mockResolvedValue(mockMetadata)
})

// ── Rendering ─────────────────────────────────────────────────────────────────

describe('Rendering', () => {
  it('1. BundleViewer renders without crashing', async () => {
    renderViewer()
    expect(document.querySelector('.viewer-wrapper')).toBeTruthy()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })

  it('2. Header shows clip_id "bg_001_b2e7f3"', async () => {
    renderViewer()
    expect(screen.getByText(/bg_001_b2e7f3/)).toBeInTheDocument()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })

  it('3. All 4 tab buttons are present with correct labels', async () => {
    renderViewer()
    expect(screen.getByRole('tab', { name: /generation record/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /edit manifest/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /integration contract/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /generation log/i })).toBeInTheDocument()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })

  it('4. "← Back to Monitor" button is present', async () => {
    renderViewer()
    expect(screen.getByRole('button', { name: /← back to monitor/i })).toBeInTheDocument()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })

  it('5. "Generation Record" tab is active by default', async () => {
    renderViewer()
    const tab = screen.getByRole('tab', { name: /generation record/i })
    expect(tab).toHaveAttribute('aria-selected', 'true')
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })

  it('6. fetchBundleFile is called on mount for metadata tab', async () => {
    renderViewer()
    await waitFor(() => {
      expect(fetchBundleFile).toHaveBeenCalledWith(
        'bg_001_b2e7f3',
        'bg_001_b2e7f3_metadata.json'
      )
    })
  })
})

// ── Tab 1 — Generation Record ─────────────────────────────────────────────────

describe('Tab 1 — Generation Record', () => {
  it('7. After metadata loads, seed value 42819 is visible', async () => {
    renderViewer()
    await waitFor(() => expect(screen.getByText('42819')).toBeInTheDocument())
  })

  it('8. User input values Economy, Urban, Dusk are visible', async () => {
    renderViewer()
    await waitFor(() => {
      expect(screen.getByText('Economy')).toBeInTheDocument()
      expect(screen.getByText('Urban')).toBeInTheDocument()
      expect(screen.getByText('Dusk')).toBeInTheDocument()
    })
  })

  it('9. Quality gate overall "pass" is visible', async () => {
    renderViewer()
    await waitFor(() => {
      // multiple "pass" occurrences may exist (luminance_gate + overall)
      const badges = screen.getAllByText('pass')
      expect(badges.length).toBeGreaterThan(0)
    })
  })

  it('10. "pass" gate result has green styling (gate-pass class)', async () => {
    renderViewer()
    await waitFor(() => {
      const badge = document.querySelector('[data-gate="pass"]')
      expect(badge).not.toBeNull()
      expect(badge.classList.contains('gate-pass')).toBe(true)
    })
  })

  it('11. positive_hash shows first 8 chars + "..."', async () => {
    renderViewer()
    await waitFor(() =>
      expect(screen.getByText('b2e7f3a1...')).toBeInTheDocument()
    )
  })

  it('12. native_fps and interpolated_to_fps shown as "8fps → 30fps"', async () => {
    renderViewer()
    await waitFor(() =>
      expect(screen.getByText('8fps → 30fps')).toBeInTheDocument()
    )
  })
})

// ── Tab 2 — Edit Manifest ─────────────────────────────────────────────────────

describe('Tab 2 — Edit Manifest', () => {
  beforeEach(() => {
    fetchBundleFile.mockImplementation((_clip_id, filename) => {
      if (filename.endsWith('_metadata.json'))      return Promise.resolve(mockMetadata)
      if (filename.endsWith('_edit_manifest.json')) return Promise.resolve(mockEditManifest)
      return Promise.resolve(null)
    })
  })

  it('13. Clicking "Edit Manifest" tab fetches edit manifest file', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /edit manifest/i }))
    await waitFor(() =>
      expect(fetchBundleFile).toHaveBeenCalledWith(
        'bg_001_b2e7f3',
        'bg_001_b2e7f3_edit_manifest.json'
      )
    )
  })

  it('14. After edit manifest loads, all 5 editable parameter names are visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /edit manifest/i }))
    await waitFor(() => {
      expect(screen.getByText('Anchor Zone')).toBeInTheDocument()
      expect(screen.getByText('Luminance Reduction')).toBeInTheDocument()
      expect(screen.getByText('LUT')).toBeInTheDocument()
      expect(screen.getByText('Clip Start Offset (s)')).toBeInTheDocument()
      expect(screen.getByText('Loop Duration Extension')).toBeInTheDocument()
    })
  })

  it('15. "extension_pass" cost renders with amber styling (cost-amber class)', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /edit manifest/i }))
    await waitFor(() => {
      const amber = document.querySelector('.cost-amber')
      expect(amber).not.toBeNull()
      expect(amber.textContent).toBe('extension_pass')
    })
  })

  it('16. Locked parameters section is visible with seed 42819', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /edit manifest/i }))
    await waitFor(() => {
      expect(screen.getByText('Locked Parameters')).toBeInTheDocument()
      expect(screen.getByText('42819')).toBeInTheDocument()
    })
  })

  it('17. Warning note text is visible in locked parameters', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /edit manifest/i }))
    await waitFor(() =>
      expect(screen.getByText(/any change requires full regeneration/i)).toBeInTheDocument()
    )
  })
})

// ── Tab 3 — Integration Contract ─────────────────────────────────────────────

describe('Tab 3 — Integration Contract', () => {
  beforeEach(() => {
    fetchBundleFile.mockImplementation((_clip_id, filename) => {
      if (filename.endsWith('_metadata.json'))              return Promise.resolve(mockMetadata)
      if (filename.endsWith('_integration_contract.json'))  return Promise.resolve(mockContract)
      return Promise.resolve(null)
    })
  })

  it('18. Clicking "Integration Contract" tab fetches contract', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() =>
      expect(fetchBundleFile).toHaveBeenCalledWith(
        'bg_001_b2e7f3',
        'bg_001_b2e7f3_integration_contract.json'
      )
    )
  })

  it('19. After contract loads, all 3 audience section headers are visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() => {
      expect(screen.getByText('For Human Editor')).toBeInTheDocument()
      expect(screen.getByText('For Downstream Modules')).toBeInTheDocument()
      expect(screen.getByText('For Compositor Process')).toBeInTheDocument()
    })
  })

  it('20. "urban_exterior" environment descriptor is visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() =>
      expect(screen.getByText('urban_exterior')).toBeInTheDocument()
    )
  })

  it('21. "minimal_dark_bar" lower_third_style is visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() =>
      expect(screen.getByText('minimal_dark_bar')).toBeInTheDocument()
    )
  })

  it('22. frame_count 512 is visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() =>
      expect(screen.getByText('512')).toBeInTheDocument()
    )
  })

  it('23. integrity_check "pass" is visible with green styling', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() => {
      const badge = document.querySelector('[data-gate="pass"]')
      expect(badge).not.toBeNull()
      expect(badge.classList.contains('gate-pass')).toBe(true)
    })
  })

  it('24. editor_notes text is visible in a callout element', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /integration contract/i }))
    await waitFor(() => {
      const callout = document.querySelector('.editor-notes-callout')
      expect(callout).not.toBeNull()
      expect(callout.textContent).toContain('Center anchor zone is pre-composited.')
    })
  })
})

// ── Tab 4 — Generation Log ────────────────────────────────────────────────────

describe('Tab 4 — Generation Log', () => {
  beforeEach(() => {
    fetchBundleFile.mockImplementation((_clip_id, filename) => {
      if (filename.endsWith('_metadata.json'))       return Promise.resolve(mockMetadata)
      if (filename.endsWith('_generation_log.json')) return Promise.resolve(mockGenerationLog)
      return Promise.resolve(null)
    })
  })

  it('25. Clicking "Generation Log" tab fetches generation log', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() =>
      expect(fetchBundleFile).toHaveBeenCalledWith(
        'bg_001_b2e7f3',
        'bg_001_b2e7f3_generation_log.json'
      )
    )
  })

  it('26. After log loads, outcome "pass_on_attempt_1" is visible', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() =>
      expect(screen.getByText('pass_on_attempt_1')).toBeInTheDocument()
    )
  })

  it('27. "No retry history" message visible when failure_log is empty', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() =>
      expect(screen.getByText(/no retry history/i)).toBeInTheDocument()
    )
  })

  it('28. escalated_to_human shown as "No"', async () => {
    renderViewer()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() => expect(screen.getByText('No')).toBeInTheDocument())
  })
})

// ── Lazy loading and caching ──────────────────────────────────────────────────

describe('Lazy loading and caching', () => {
  it('29. Switching to a tab already fetched does not call fetchBundleFile again', async () => {
    fetchBundleFile.mockImplementation((_clip_id, filename) => {
      if (filename.endsWith('_metadata.json'))       return Promise.resolve(mockMetadata)
      if (filename.endsWith('_generation_log.json')) return Promise.resolve(mockGenerationLog)
      return Promise.resolve(null)
    })

    renderViewer()
    // Wait for metadata fetch on mount
    await waitFor(() =>
      expect(fetchBundleFile).toHaveBeenCalledWith('bg_001_b2e7f3', 'bg_001_b2e7f3_metadata.json')
    )

    // Navigate to log tab (fetches)
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() =>
      expect(fetchBundleFile).toHaveBeenCalledWith('bg_001_b2e7f3', 'bg_001_b2e7f3_generation_log.json')
    )
    const countAfterFirstVisit = fetchBundleFile.mock.calls.length

    // Go back to metadata (already cached — must NOT re-fetch)
    await userEvent.click(screen.getByRole('tab', { name: /generation record/i }))
    // Small pause to confirm no extra call happens
    await new Promise((r) => setTimeout(r, 50))
    expect(fetchBundleFile.mock.calls.length).toBe(countAfterFirstVisit)
  })
})

// ── Error handling ────────────────────────────────────────────────────────────

describe('Error handling', () => {
  it('30. When fetchBundleFile returns null (404), "File not available" is shown', async () => {
    fetchBundleFile.mockResolvedValue(null)
    renderViewer()
    await waitFor(() =>
      expect(screen.getByText(/file not available for this run/i)).toBeInTheDocument()
    )
  })

  it('31. When fetchBundleFile throws, error message shown; other tabs unaffected', async () => {
    // Metadata throws, but other tabs should still be navigable
    fetchBundleFile.mockImplementation((_clip_id, filename) => {
      if (filename.endsWith('_metadata.json'))
        return Promise.reject(new Error('Bundle fetch failed: 500'))
      return Promise.resolve(mockGenerationLog)
    })

    renderViewer()

    // Metadata tab should show unavailable (error path → null state)
    await waitFor(() =>
      expect(screen.getByText(/file not available for this run/i)).toBeInTheDocument()
    )

    // Navigate to generation log — should work fine
    await userEvent.click(screen.getByRole('tab', { name: /generation log/i }))
    await waitFor(() =>
      expect(screen.getByText('pass_on_attempt_1')).toBeInTheDocument()
    )
  })
})

// ── Back navigation ───────────────────────────────────────────────────────────

describe('Back navigation', () => {
  it('32. Clicking "← Back to Monitor" calls onBack prop', async () => {
    const onBack = vi.fn()
    renderViewer({ onBack })
    await userEvent.click(screen.getByRole('button', { name: /← back to monitor/i }))
    expect(onBack).toHaveBeenCalled()
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
  })
})

// ── App.jsx three-screen flow ────────────────────────────────────────────────

const mockCompileResult = {
  input_hash_short:  'b2e7f3',
  selected_lut:      'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  positive_hash:     'b2e7f3a1c9d4e8f2',
  motion_hash:       '9fa14c82',
  negative_hash:     'c4d8b901',
}

const mockRunResultComplete = {
  run_id:               'bg_001_b2e7f3',
  status:               'complete',
  raw_loop_path:        'raw/bg_001_b2e7f3_raw_loop.mp4',
  seed:                 42819,
  seam_frames_raw:      [183, 366],
  seam_frames_playable: [169, 338],
  stages:               {},
  gate_result:          { overall: 'pass' },
}

async function fillAndCompile(user) {
  const selects = screen.getAllByRole('combobox')
  const fields = [
    'category', 'location_feel', 'time_of_day',
    'color_temperature', 'mood', 'motion_intensity',
  ]
  for (let i = 0; i < fields.length; i++) {
    await user.selectOptions(selects[i], FORM_OPTIONS[fields[i]][0].value)
  }
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
}

describe('App.jsx three-screen flow', () => {
  it('33. App renders EditorialForm initially', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
  })

  it('34. After onCompileSuccess, App renders RunMonitor', async () => {
    compilePrompts.mockResolvedValue(mockCompileResult)
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue({ status: 'running', stages: {} })

    const user = userEvent.setup()
    render(<App />)
    await fillAndCompile(user)

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /compile prompts/i })).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: /← new run/i })).toBeInTheDocument()
    })
  })

  // Tests 35 and 36 exercise the full polling cycle.
  // RunMonitor polls every 2s; per-test Vitest timeout is extended to 10s
  // and waitFor timeout to 5s so the poll has time to fire.

  it('35. After RunMonitor onComplete fires, App renders BundleViewer', async () => {
    compilePrompts.mockResolvedValue(mockCompileResult)
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue(mockRunResultComplete)
    fetchBundleFile.mockResolvedValue(mockMetadata)

    const user = userEvent.setup()
    render(<App />)
    await fillAndCompile(user)

    // When the poll returns 'complete', onComplete auto-fires → screen transitions to 'viewer'
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /← back to monitor/i })).toBeInTheDocument(),
      { timeout: 5000 }
    )
    // clip_id appears in both the viewer header and the Clip ID InfoRow
    expect(screen.getAllByText(/bg_001_b2e7f3/).length).toBeGreaterThan(0)
  }, 10000)

  it('36. BundleViewer onBack returns to RunMonitor without re-running', async () => {
    compilePrompts.mockResolvedValue(mockCompileResult)
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue(mockRunResultComplete)
    fetchBundleFile.mockResolvedValue(mockMetadata)

    const user = userEvent.setup()
    render(<App />)
    await fillAndCompile(user)

    // Wait for automatic transition to viewer (poll fires at ~2s)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /← back to monitor/i })).toBeInTheDocument(),
      { timeout: 5000 }
    )

    const submitCallsBefore = submitGeneration.mock.calls.length
    await user.click(screen.getByRole('button', { name: /← back to monitor/i }))

    // Should return to RunMonitor (← New Run button visible, not EditorialForm)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /← new run/i })).toBeInTheDocument()
    )

    // submitGeneration must NOT have been called again (no re-run)
    expect(submitGeneration.mock.calls.length).toBe(submitCallsBefore)

    // EditorialForm must not be visible (still on monitor, not form)
    expect(screen.queryByRole('button', { name: /compile prompts/i })).not.toBeInTheDocument()
  }, 10000)
})
