import React from 'react'
import { render, screen, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import RunMonitor from './RunMonitor.jsx'
import App from '../App.jsx'
import FORM_OPTIONS from '../config/formOptions.js'

vi.mock('../api/client', () => ({
  submitGeneration: vi.fn(),
  getRunStatus:     vi.fn(),
  compilePrompts:   vi.fn(),
}))

import { submitGeneration, getRunStatus, compilePrompts } from '../api/client'

const mockCompileResult = {
  input_hash_short:  'b2e7f3',
  selected_lut:      'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  positive_hash:     'b2e7f3a1c9d4e8f2',
  motion_hash:       '9fa14c82',
  negative_hash:     'c4d8b901',
}

const mockFormData = {
  category:          'Economy',
  location_feel:     'Urban',
  time_of_day:       'Dusk',
  color_temperature: 'Cool',
  mood:              'Serious',
  motion_intensity:  'Gentle',
}

const mockRunResult = {
  run_id:               'bg_001_b2e7f3',
  status:               'complete',
  raw_loop_path:        'raw/bg_001_b2e7f3_raw_loop.mp4',
  seed:                 42819,
  seam_frames_raw:      [183, 366],
  seam_frames_playable: [138, 269],
  stages: {
    prompt_compilation: 'complete',
    generation:         'complete',
    probe_decode:       'complete',
    probe_temporal:     'complete',
    gate_evaluation:    'complete',
    upscale:            'complete',
    mask_generation:    'complete',
    lut_grading:        'complete',
    composite:          'complete',
    preview_export:     'complete',
    metadata_assembly:  'complete',
  },
  gate_result: { overall: 'pass' },
}

const defaultProps = {
  compileResult: mockCompileResult,
  formData:      mockFormData,
  onBack:        vi.fn(),
  onComplete:    vi.fn(),
}

function renderMonitor(props = {}) {
  return render(<RunMonitor {...defaultProps} {...props} />)
}

beforeEach(() => {
  vi.clearAllMocks()
  submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
  getRunStatus.mockResolvedValue({ status: 'running', stages: {} })
})

afterEach(() => {
  vi.useRealTimers()
})

// Helper: render monitor with fake timers, flush submitGeneration, advance to first poll
async function renderAndPoll(statusResponse, extraProps = {}) {
  vi.useFakeTimers()
  submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
  getRunStatus.mockResolvedValue(statusResponse)

  const result = renderMonitor(extraProps)

  // Flush submitGeneration promise and resulting state updates
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })

  // Advance past the 2s poll interval and flush the async callback
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2001)
  })

  // Flush any remaining state updates
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })

  return result
}

// ── Rendering on mount ───────────────────────────────────────────────────────

describe('Rendering on mount', () => {
  it('1. RunMonitor renders without crashing', async () => {
    renderMonitor()
    expect(document.querySelector('.monitor-wrapper')).toBeTruthy()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('2. Run ID is displayed', async () => {
    renderMonitor()
    expect(screen.getByText(/b2e7f3/)).toBeInTheDocument()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('3. All 11 stage labels are present', async () => {
    renderMonitor()
    expect(screen.getByText('Prompt Compilation')).toBeInTheDocument()
    expect(screen.getByText('Video Generation')).toBeInTheDocument()
    expect(screen.getByText('Decode Probe')).toBeInTheDocument()
    expect(screen.getByText('Temporal Probe')).toBeInTheDocument()
    expect(screen.getByText('Quality Gates')).toBeInTheDocument()
    expect(screen.getByText('Upscale 1080p')).toBeInTheDocument()
    expect(screen.getByText('Mask Generation')).toBeInTheDocument()
    expect(screen.getByText('LUT Grading')).toBeInTheDocument()
    expect(screen.getByText('Composite')).toBeInTheDocument()
    expect(screen.getByText('Preview Export')).toBeInTheDocument()
    expect(screen.getByText('Metadata Assembly')).toBeInTheDocument()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('4. "← New Run" back button is present', async () => {
    renderMonitor()
    expect(screen.getByRole('button', { name: /← new run/i })).toBeInTheDocument()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('5. All 6 editorial input values appear in the run summary strip', async () => {
    renderMonitor()
    expect(screen.getByText('Economy')).toBeInTheDocument()
    expect(screen.getByText('Urban')).toBeInTheDocument()
    expect(screen.getByText('Dusk')).toBeInTheDocument()
    expect(screen.getByText('Cool')).toBeInTheDocument()
    expect(screen.getByText('Serious')).toBeInTheDocument()
    expect(screen.getByText('Gentle')).toBeInTheDocument()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('6. No result panel visible on initial render', () => {
    renderMonitor()
    expect(screen.queryByText('Run Result')).not.toBeInTheDocument()
  })

  it('7. submitGeneration is called once on mount', async () => {
    renderMonitor()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalledTimes(1))
  })
})

// ── Running state ────────────────────────────────────────────────────────────

describe('Running state', () => {
  it('8. While submitGeneration is pending, loading/initializing state is visible', async () => {
    let resolveSubmit
    submitGeneration.mockReturnValue(
      new Promise((r) => { resolveSubmit = r })
    )

    renderMonitor()
    expect(screen.getByText(/initializing/i)).toBeInTheDocument()

    await act(async () => {
      resolveSubmit({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
      await Promise.resolve()
    })
  })

  it('9. After submitGeneration resolves, polling begins (getRunStatus is called)', async () => {
    vi.useFakeTimers()
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue({ status: 'running', stages: {} })

    renderMonitor()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2001)
    })

    expect(getRunStatus).toHaveBeenCalled()
  })
})

// ── Stage status updates ─────────────────────────────────────────────────────

describe('Stage status updates', () => {
  it('10. When getRunStatus returns generation: "running", Video Generation row shows running state', async () => {
    await renderAndPoll({ status: 'running', stages: { generation: 'running' } })

    const row = screen.getByTestId('stage-generation')
    expect(row).toHaveAttribute('data-status', 'running')
  })

  it('11. When getRunStatus returns all stages complete, all 11 stage rows show complete state', async () => {
    await renderAndPoll(mockRunResult)

    const stageKeys = [
      'prompt_compilation', 'generation', 'probe_decode', 'probe_temporal',
      'gate_evaluation', 'upscale', 'mask_generation', 'lut_grading',
      'composite', 'preview_export', 'metadata_assembly',
    ]
    for (const key of stageKeys) {
      expect(screen.getByTestId(`stage-${key}`)).toHaveAttribute('data-status', 'complete')
    }
  })
})

// ── Terminal states ──────────────────────────────────────────────────────────

describe('Terminal states', () => {
  it('12. When status == "complete", result panel appears', async () => {
    await renderAndPoll(mockRunResult)
    expect(screen.getByText('Run Result')).toBeInTheDocument()
  })

  it('13. Result panel shows seed value 42819', async () => {
    await renderAndPoll(mockRunResult)
    expect(screen.getByText('42819')).toBeInTheDocument()
  })

  it('14. Result panel shows gate result "pass"', async () => {
    await renderAndPoll(mockRunResult)
    expect(screen.getByText('pass')).toBeInTheDocument()
  })

  it('15. "View Output Bundle →" button is present and enabled when status == "complete"', async () => {
    await renderAndPoll(mockRunResult)
    const btn = screen.getByRole('button', { name: /view output bundle/i })
    expect(btn).toBeInTheDocument()
    expect(btn).not.toBeDisabled()
  })

  it('16. onComplete prop is called when status == "complete"', async () => {
    const onComplete = vi.fn()
    await renderAndPoll(mockRunResult, { onComplete })
    expect(onComplete).toHaveBeenCalledWith(mockRunResult)
  })

  it('17. When status == "failed", error banner is visible with "failed" text', async () => {
    await renderAndPoll({ run_id: 'bg_001_b2e7f3', status: 'failed', stages: {} })
    expect(screen.getByText(/run failed/i)).toBeInTheDocument()
  })

  it('18. When status == "escalated", amber/escalated banner is visible', async () => {
    await renderAndPoll({ run_id: 'bg_001_b2e7f3', status: 'escalated', stages: {} })
    expect(screen.getByText(/escalated to human review/i)).toBeInTheDocument()
  })
})

// ── Polling cleanup ──────────────────────────────────────────────────────────

describe('Polling cleanup', () => {
  it('19. getRunStatus is not called after terminal state is reached', async () => {
    await renderAndPoll(mockRunResult)

    const callsAfterTerminal = getRunStatus.mock.calls.length

    // Advance more time — polling is stopped, so no new calls
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(getRunStatus.mock.calls.length).toBe(callsAfterTerminal)
  })

  it('20. Polling interval is cleared on unmount', async () => {
    vi.useFakeTimers()
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue({ status: 'running', stages: {} })

    const { unmount } = renderMonitor()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2001)
    })

    const callsBeforeUnmount = getRunStatus.mock.calls.length
    expect(callsBeforeUnmount).toBeGreaterThan(0)

    unmount()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000)
    })

    expect(getRunStatus.mock.calls.length).toBe(callsBeforeUnmount)
  })
})

// ── Back navigation ──────────────────────────────────────────────────────────

describe('Back navigation', () => {
  it('21. Clicking "← New Run" calls onBack prop', async () => {
    const onBack = vi.fn()
    renderMonitor({ onBack })
    await userEvent.click(screen.getByRole('button', { name: /← new run/i }))
    expect(onBack).toHaveBeenCalled()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })

  it('22. onBack is called with no arguments', async () => {
    const onBack = vi.fn()
    renderMonitor({ onBack })
    await userEvent.click(screen.getByRole('button', { name: /← new run/i }))
    expect(onBack).toHaveBeenCalledWith()
    await waitFor(() => expect(submitGeneration).toHaveBeenCalled())
  })
})

// ── Error handling ───────────────────────────────────────────────────────────

describe('Error handling', () => {
  it('23. When submitGeneration rejects, error message appears with role="alert"', async () => {
    submitGeneration.mockRejectedValue(new Error('Generation failed: 503 Service Unavailable'))
    renderMonitor()
    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
      expect(screen.getByText(/Generation failed/i)).toBeInTheDocument()
    })
  })

  it('24. When getRunStatus rejects mid-run, error message appears with role="alert"', async () => {
    vi.useFakeTimers()
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockRejectedValue(new Error('Status check failed: 500'))

    renderMonitor()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2001)
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(screen.getByRole('alert')).toBeInTheDocument()
  })
})

// ── Elapsed timer ────────────────────────────────────────────────────────────

describe('Elapsed timer', () => {
  it('25. Elapsed time display is present in document', () => {
    renderMonitor()
    expect(screen.getByText('0:00')).toBeInTheDocument()
  })

  it('26. After 65 seconds elapsed, display shows "1:05"', async () => {
    vi.useFakeTimers()
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue({ status: 'running', stages: {} })

    renderMonitor()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(65000)
    })

    expect(screen.getByText('1:05')).toBeInTheDocument()
  })
})

// ── EditorialForm / App.jsx integration ─────────────────────────────────────

describe('EditorialForm App.jsx integration', () => {
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

  it('27. App renders EditorialForm initially', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
  })

  it('28. After EditorialForm calls onCompileSuccess, App renders RunMonitor (not EditorialForm)', async () => {
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

  it('29. Clicking "← New Run" in RunMonitor causes App to render EditorialForm again', async () => {
    compilePrompts.mockResolvedValue(mockCompileResult)
    submitGeneration.mockResolvedValue({ run_id: 'bg_001_b2e7f3', status: 'running', stages: {} })
    getRunStatus.mockResolvedValue({ status: 'running', stages: {} })

    const user = userEvent.setup()
    render(<App />)
    await fillAndCompile(user)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /← new run/i })).toBeInTheDocument()
    )

    await user.click(screen.getByRole('button', { name: /← new run/i }))

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
    )
  })
})
