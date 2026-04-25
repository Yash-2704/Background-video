import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import PrototypeViewer from './PrototypeViewer.jsx'

const SUCCESS_RESPONSE = {
  run_id: 'abc12345',
  image_path: 'output/prototype/abc12345/image.png',
  video_path: 'output/prototype/abc12345/animated.mp4',
  prompt_used: 'a vivid scene',
}

function makeFetchSuccess(data) {
  return vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve(data),
  })
}

beforeEach(() => {
  vi.restoreAllMocks()
})

// 1. renders_heading
it('renders_heading', () => {
  render(<PrototypeViewer />)
  expect(screen.getByText('Prototype Generator')).toBeInTheDocument()
})

// 2. renders_all_six_form_fields
it('renders_all_six_form_fields', () => {
  render(<PrototypeViewer />)
  expect(screen.getByText('Category')).toBeInTheDocument()
  expect(screen.getByText('Location Feel')).toBeInTheDocument()
  expect(screen.getByText('Time of Day')).toBeInTheDocument()
  expect(screen.getByText('Color Temperature')).toBeInTheDocument()
  expect(screen.getByText('Mood')).toBeInTheDocument()
  // Motion Intensity label includes the current value — use a regex
  expect(screen.getByText(/Motion Intensity/)).toBeInTheDocument()
})

// 3. submit_button_present_and_enabled_by_default
it('submit_button_present_and_enabled_by_default', () => {
  render(<PrototypeViewer />)
  const btn = screen.getByRole('button', { name: /generate/i })
  expect(btn).toBeInTheDocument()
  expect(btn).not.toBeDisabled()
})

// 4. submit_button_disabled_while_loading
it('submit_button_disabled_while_loading', async () => {
  vi.stubGlobal('fetch', vi.fn().mockReturnValue(new Promise(() => {}))) // never resolves
  const user = userEvent.setup()
  render(<PrototypeViewer />)
  await user.click(screen.getByRole('button', { name: /generate/i }))
  expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled()
})

// 5. shows_error_on_bad_response
it('shows_error_on_bad_response', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: false,
    json: () => Promise.resolve({ detail: 'Something broke' }),
  }))
  const user = userEvent.setup()
  render(<PrototypeViewer />)
  await user.click(screen.getByRole('button', { name: /generate/i }))
  expect(await screen.findByText('Something broke')).toBeInTheDocument()
})

// 6. shows_result_video_on_success
it('shows_result_video_on_success', async () => {
  vi.stubGlobal('fetch', makeFetchSuccess(SUCCESS_RESPONSE))
  const user = userEvent.setup()
  render(<PrototypeViewer />)
  await user.click(screen.getByRole('button', { name: /generate/i }))
  const video = await screen.findByRole('figure', { hidden: true }) // fallback: query by tag
    .catch(() => document.querySelector('video'))
  // Directly check DOM since <video> has no implicit ARIA role
  await waitFor(() => {
    const videoEl = document.querySelector('video')
    expect(videoEl).toBeTruthy()
    expect(videoEl.getAttribute('src')).toContain('animated.mp4')
  })
})

// 7. shows_prompt_used_on_success
it('shows_prompt_used_on_success', async () => {
  vi.stubGlobal('fetch', makeFetchSuccess(SUCCESS_RESPONSE))
  const user = userEvent.setup()
  render(<PrototypeViewer />)
  await user.click(screen.getByRole('button', { name: /generate/i }))
  expect(await screen.findByText('a vivid scene')).toBeInTheDocument()
})

// 8. motion_intensity_slider_default_value
it('motion_intensity_slider_default_value', () => {
  render(<PrototypeViewer />)
  const slider = document.querySelector('input[type="range"]')
  expect(slider).toBeTruthy()
  expect(slider.value).toBe('0.5')
})
