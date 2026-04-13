import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import BundleViewer from './BundleViewer.jsx'

// ── Mock API client ───────────────────────────────────────────────────────────

vi.mock('../api/client', () => ({
  submitGeneration: vi.fn(),
  getRunStatus:     vi.fn(),
  compilePrompts:   vi.fn(),
  fetchBundleFile:  vi.fn(),
  getMediaUrl: (clip_id, filename) =>
    `http://localhost:8000/api/v1/media/${clip_id}/${filename}`,
}))

import { fetchBundleFile } from '../api/client'

// ── Fixtures ──────────────────────────────────────────────────────────────────

const mockMetadataWithVideo = {
  clip_id:        'bg_001_b2e7f3',
  module_version: '1.1',
  schema_version: '1.0',
  timestamp_utc:  '2026-04-06T09:14:00Z',
  dev_mode:       true,
  generation: {
    model: 'CogVideoX-5b', sampler: 'DPM++ 2M',
    cfg_scale: 6.0, steps: 50, seed: 42819,
    native_fps: 24,
    base_duration_s: 6.04, extensions: 2,
    total_raw_duration_s: 18.12, playable_duration_s: 16.917,
    crossfade_frames: 14,
    seam_frames_playable: [138, 269],
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
    raw_loop:       'raw/bg_001_b2e7f3_raw_loop.mp4',
    upscaled:       'raw/bg_001_b2e7f3_1080p.mp4',
    decode_probe:   'raw/bg_001_b2e7f3_decode_probe.json',
    temporal_probe: 'raw/bg_001_b2e7f3_temporal_probe.json',
    final:   'final/bg_001_b2e7f3.mp4',
    preview: 'final/bg_001_b2e7f3_preview.gif',
    masks: [], luts: [],
  },
}

const mockRunResult = {
  run_id:               'bg_001_b2e7f3',
  status:               'complete',
  raw_loop_path:        'raw/bg_001_b2e7f3_raw_loop.mp4',
  seed:                 42819,
  seam_frames_playable: [138, 269],
  gate_result:          { overall: 'pass' },
}

const defaultProps = {
  runResult: mockRunResult,
  onBack:    vi.fn(),
}

function renderViewer(metadataOverride) {
  fetchBundleFile.mockResolvedValue(metadataOverride ?? mockMetadataWithVideo)
  return render(<BundleViewer {...defaultProps} />)
}

beforeEach(() => {
  vi.clearAllMocks()
})

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('BundleViewer video playback', () => {
  it('test_01: video element is present when metadata has files.final', async () => {
    renderViewer()
    await waitFor(() =>
      expect(screen.getByTestId('final-video-player')).toBeInTheDocument()
    )
  })

  it('test_02: video src is the correct media URL', async () => {
    renderViewer()
    await waitFor(() => {
      const video = screen.getByTestId('final-video-player')
      expect(video).toHaveAttribute(
        'src',
        'http://localhost:8000/api/v1/media/bg_001_b2e7f3/bg_001_b2e7f3.mp4'
      )
    })
  })

  it('test_03: video element has controls attribute', async () => {
    renderViewer()
    await waitFor(() => {
      const video = screen.getByTestId('final-video-player')
      expect(video).toHaveAttribute('controls')
    })
  })

  it('test_04: video element has loop attribute', async () => {
    renderViewer()
    await waitFor(() => {
      const video = screen.getByTestId('final-video-player')
      expect(video).toHaveAttribute('loop')
    })
  })

  it('test_05: img with data-testid="preview-gif" is present when metadata has files.preview', async () => {
    renderViewer()
    await waitFor(() =>
      expect(screen.getByTestId('preview-gif')).toBeInTheDocument()
    )
  })

  it('test_06: gif src is the correct media URL', async () => {
    renderViewer()
    await waitFor(() => {
      const gif = screen.getByTestId('preview-gif')
      expect(gif).toHaveAttribute(
        'src',
        'http://localhost:8000/api/v1/media/bg_001_b2e7f3/bg_001_b2e7f3_preview.gif'
      )
    })
  })

  it('test_07: video player does NOT render when files.final is "MISSING"', async () => {
    const meta = {
      ...mockMetadataWithVideo,
      files: { ...mockMetadataWithVideo.files, final: 'MISSING' },
    }
    renderViewer(meta)
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    // Allow a tick for state to settle
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByTestId('final-video-player')).not.toBeInTheDocument()
  })

  it('test_08: video player does NOT render when files section is absent', async () => {
    const { files: _omit, ...metaWithoutFiles } = mockMetadataWithVideo
    renderViewer(metaWithoutFiles)
    await waitFor(() => expect(fetchBundleFile).toHaveBeenCalled())
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.queryByTestId('final-video-player')).not.toBeInTheDocument()
  })

  it('test_09: "Final Composite" heading is visible when video player renders', async () => {
    renderViewer()
    await waitFor(() =>
      expect(screen.getByText('Final Composite')).toBeInTheDocument()
    )
  })

  it('test_10: preview label contains clip_id, duration_s, and fps values', async () => {
    renderViewer()
    await waitFor(() => {
      const label = document.querySelector('.video-preview-label')
      expect(label).not.toBeNull()
      expect(label.textContent).toContain('bg_001_b2e7f3')
      expect(label.textContent).toContain('16.917')
      expect(label.textContent).toContain('24')
    })
  })
})
