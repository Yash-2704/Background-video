import React from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '../App.jsx'

vi.mock('../api/client', () => ({
  compilePrompts: vi.fn(),
  submitGeneration: vi.fn(),
  getRunStatus: vi.fn(),
  fetchBundleFile: vi.fn(),
}))

import { compilePrompts, fetchBundleFile, getRunStatus, submitGeneration } from '../api/client'

const canonicalForm = {
  category: 'Economy',
  location_feel: 'Urban',
  time_of_day: 'Dusk',
  color_temperature: 'Cool',
  mood: 'Serious',
  motion_intensity: 'Gentle',
}

const allStagesComplete = {
  prompt_compilation: 'complete',
  generation: 'complete',
  probe_decode: 'complete',
  probe_temporal: 'complete',
  gate_evaluation: 'complete',
  upscale: 'complete',
  mask_generation: 'complete',
  lut_grading: 'complete',
  composite: 'complete',
  preview_export: 'complete',
  metadata_assembly: 'complete',
}

const mockCompileResponse = {
  positive: 'pos',
  motion: 'mot',
  negative: 'neg',
  positive_hash: 'a1b2c3d4',
  motion_hash: 'b2c3d4e5',
  negative_hash: 'c3d4e5f6',
  input_hash_short: 'abc123',
  selected_lut: 'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  compiler_version: '1.0.0',
  user_input: canonicalForm,
}

const mockGenerateResponse = {
  run_id: 'abc123',
  status: 'complete',
  raw_loop_path: 'raw/abc123_raw_loop.mp4',
  seed: 55123,
  seam_frames_raw: [183, 366],
  seam_frames_playable: [169, 338],
  gate_result: { overall: 'pass' },
  selected_lut: 'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  metadata_path: 'output/abc123/abc123_metadata.json',
  stages: allStagesComplete,
}

const mockStatusResponse = {
  run_id: 'abc123',
  status: 'complete',
  stages: allStagesComplete,
}

const mockMetadataFile = {
  clip_id: 'abc123',
  module_version: '1.1',
  schema_version: '1.0',
  timestamp_utc: '2026-01-01T00:00:00Z',
  dev_mode: true,
  generation: { native_fps: 8, interpolated_to_fps: 30, seed: 55123, seam_frames_playable: [169, 338] },
  prompt_provenance: { input_hash_short: 'abc123', positive_hash: 'a1b2c3d4', compiler_version: '1.0.0' },
  user_input: canonicalForm,
  quality_gates: { overall: 'pass' },
  post_processing: { selected_lut: 'cool_authority', luts_generated: [], masks_generated: [] },
}

async function fillEditorialForm(user) {
  const selects = screen.getAllByRole('combobox')
  await user.selectOptions(selects[0], canonicalForm.category)
  await user.selectOptions(selects[1], canonicalForm.location_feel)
  await user.selectOptions(selects[2], canonicalForm.time_of_day)
  await user.selectOptions(selects[3], canonicalForm.color_temperature)
  await user.selectOptions(selects[4], canonicalForm.mood)
  await user.selectOptions(selects[5], canonicalForm.motion_intensity)
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
  compilePrompts.mockResolvedValue(mockCompileResponse)
  submitGeneration.mockResolvedValue(mockGenerateResponse)
  getRunStatus.mockResolvedValue(mockStatusResponse)
  fetchBundleFile.mockResolvedValue(mockMetadataFile)
})

it('1. App renders EditorialForm on initial load', () => {
  render(<App />)
  expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
})

it('2. RunMonitor is not visible on initial load', () => {
  render(<App />)
  expect(screen.queryByRole('button', { name: /← new run/i })).not.toBeInTheDocument()
})

it('3. BundleViewer is not visible on initial load', () => {
  render(<App />)
  expect(screen.queryByRole('button', { name: /← back to monitor/i })).not.toBeInTheDocument()
})

it('4. Completing form and clicking compile calls compilePrompts once', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await waitFor(() => expect(compilePrompts).toHaveBeenCalledTimes(1))
})

it('5. After compile resolves, RunMonitor becomes visible', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByRole('button', { name: /← new run/i })).toBeInTheDocument()
})

it('6. EditorialForm is hidden after form to monitor transition', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← new run/i })
  expect(screen.queryByRole('button', { name: /compile prompts/i })).not.toBeInTheDocument()
})

it('7. Run ID from compileResult appears in RunMonitor', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByText(/abc123/)).toBeInTheDocument()
})

it('8. RunMonitor calls submitGeneration on mount', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await waitFor(() => expect(submitGeneration).toHaveBeenCalledTimes(1))
})

it('9. Complete status transition shows BundleViewer', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })).toBeInTheDocument()
})

it('10. RunMonitor stays mounted but hidden when viewer visible', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  const monitorRoot = document.querySelector('.monitor-wrapper')
  expect(monitorRoot).toBeInTheDocument()
  expect(monitorRoot).not.toBeVisible()
})

it('11. BundleViewer header shows clip_id', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  expect(screen.getByText(/output bundle — abc123/i)).toBeInTheDocument()
})

it('12. BundleViewer renders all four tab buttons', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  expect(screen.getByRole('tab', { name: /generation record/i })).toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /edit manifest/i })).toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /integration contract/i })).toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /generation log/i })).toBeInTheDocument()
})

it('13. BundleViewer mount triggers fetchBundleFile', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  await waitFor(() =>
    expect(fetchBundleFile).toHaveBeenCalledWith('abc123', 'abc123_metadata.json')
  )
})

it('14. Back to Monitor hides viewer and shows monitor', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  await user.click(screen.getByRole('button', { name: /← back to monitor/i }))
  expect(screen.queryByRole('button', { name: /← back to monitor/i })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: /← new run/i })).toBeVisible()
})

it('15. Returning to monitor does not re-submit generation', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  const callsBefore = submitGeneration.mock.calls.length
  await user.click(screen.getByRole('button', { name: /← back to monitor/i }))
  expect(submitGeneration.mock.calls.length).toBe(callsBefore)
})

it('16. Clicking New Run returns to EditorialForm and hides monitor', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  await screen.findByRole('button', { name: /← new run/i })
  await user.click(screen.getByRole('button', { name: /← new run/i }))
  expect(await screen.findByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /← new run/i })).not.toBeInTheDocument()
})

it('17. Editorial inputs appear in RunMonitor summary strip', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByText('Economy')).toBeInTheDocument()
  expect(screen.getByText('Urban')).toBeInTheDocument()
  expect(screen.getByText('Dusk')).toBeInTheDocument()
  expect(screen.getByText('Cool')).toBeInTheDocument()
  expect(screen.getByText('Serious')).toBeInTheDocument()
  expect(screen.getByText('Gentle')).toBeInTheDocument()
})

it('18. Same run_id appears in monitor and viewer headers', async () => {
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByText(/run: abc123/i)).toBeInTheDocument()
  await screen.findByRole('button', { name: /← back to monitor/i }, { timeout: 8000 })
  expect(screen.getByText(/output bundle — abc123/i)).toBeInTheDocument()
})

it('19. compilePrompts rejection keeps app on EditorialForm and shows error', async () => {
  compilePrompts.mockRejectedValueOnce(new Error('compile failed'))
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByRole('alert')).toHaveTextContent('compile failed')
  expect(screen.queryByRole('button', { name: /← new run/i })).not.toBeInTheDocument()
})

it('20. submitGeneration rejection keeps app on RunMonitor with error', async () => {
  submitGeneration.mockRejectedValueOnce(new Error('generation failed'))
  const user = userEvent.setup()
  render(<App />)
  await fillEditorialForm(user)
  await user.click(screen.getByRole('button', { name: /compile prompts/i }))
  expect(await screen.findByRole('alert')).toHaveTextContent('generation failed')
  expect(screen.queryByRole('button', { name: /← back to monitor/i })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: /← new run/i })).toBeInTheDocument()
})
