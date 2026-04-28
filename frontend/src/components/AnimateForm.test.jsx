import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import AnimateForm from './AnimateForm.jsx'

vi.mock('../api/client', () => ({
  uploadImage:     vi.fn(),
  submitAnimation: vi.fn(),
  API_BASE:        'http://test-host:8001',
}))

import { uploadImage, submitAnimation } from '../api/client'

const mockUploadResult = {
  image_id:          'img-uuid-001',
  preview_url:       '/api/v1/preview/img-uuid-001.jpg',
  width:             1920,
  height:            1080,
  original_filename: 'background.jpg',
}

const mockAnimResult = {
  run_id:            'abc123',
  input_hash_short:  'abc123',
  status:            'running',
}

const testFile = new File(['dummy'], 'bg.jpg', { type: 'image/jpeg' })

beforeEach(() => {
  vi.clearAllMocks()
  global.URL.createObjectURL = vi.fn(() => 'blob:mock-url')
  global.URL.revokeObjectURL = vi.fn()
})

// ── Helper: select a file ────────────────────────────────────────────────────

async function selectFile(user) {
  const input = document.querySelector('input[type="file"]')
  await user.upload(input, testFile)
}

// ── Helper: advance through full upload flow ─────────────────────────────────

async function reachConfirm(user, prompt = 'Light streaks left to right') {
  uploadImage.mockResolvedValueOnce(mockUploadResult)
  await selectFile(user)
  await user.type(screen.getByRole('textbox'), prompt)
  await user.click(screen.getByRole('button', { name: /animate →/i }))
  await screen.findByText('Ready to animate')
}

// ── TEST A ───────────────────────────────────────────────────────────────────

describe('TEST A — renders file input and textarea on load', () => {
  it('file input and textarea are present', () => {
    render(<AnimateForm />)
    expect(document.querySelector('input[type="file"]')).toBeInTheDocument()
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })
})

// ── TEST B ───────────────────────────────────────────────────────────────────

describe('TEST B — Animate button disabled when no file selected', () => {
  it('button stays disabled after typing prompt with no file', async () => {
    const user = userEvent.setup()
    render(<AnimateForm />)
    await user.type(screen.getByRole('textbox'), 'some prompt')
    expect(screen.getByRole('button', { name: /animate →/i })).toBeDisabled()
  })
})

// ── TEST C ───────────────────────────────────────────────────────────────────

describe('TEST C — Animate button disabled when no prompt', () => {
  it('button stays disabled after file selection with no prompt', async () => {
    const user = userEvent.setup()
    render(<AnimateForm />)
    await selectFile(user)
    expect(screen.getByRole('button', { name: /animate →/i })).toBeDisabled()
  })
})

// ── TEST D ───────────────────────────────────────────────────────────────────

describe('TEST D — clicking Animate calls uploadImage', () => {
  it('uploadImage is called with the selected file', async () => {
    uploadImage.mockResolvedValueOnce(mockUploadResult)
    const user = userEvent.setup()
    render(<AnimateForm />)
    await selectFile(user)
    await user.type(screen.getByRole('textbox'), 'shimmer on panels')
    await user.click(screen.getByRole('button', { name: /animate →/i }))
    await waitFor(() => expect(uploadImage).toHaveBeenCalledWith(testFile))
  })
})

// ── TEST E ───────────────────────────────────────────────────────────────────

describe('TEST E — after upload, confirm step appears', () => {
  it('"Ready to animate" heading is visible', async () => {
    const user = userEvent.setup()
    render(<AnimateForm />)
    await reachConfirm(user)
    expect(screen.getByText('Ready to animate')).toBeInTheDocument()
  })
})

// ── TEST F ───────────────────────────────────────────────────────────────────

describe('TEST F — Edit button returns to upload step', () => {
  it('file input is visible again after clicking ← Edit', async () => {
    const user = userEvent.setup()
    render(<AnimateForm />)
    await reachConfirm(user)
    await user.click(screen.getByRole('button', { name: /← edit/i }))
    expect(document.querySelector('input[type="file"]')).toBeInTheDocument()
  })
})

// ── TEST G ───────────────────────────────────────────────────────────────────

describe('TEST G — Start Animation calls submitAnimation', () => {
  it('submitAnimation is called after clicking Start Animation', async () => {
    submitAnimation.mockResolvedValueOnce(mockAnimResult)
    const user = userEvent.setup()
    render(<AnimateForm onAnimateSuccess={vi.fn()} />)
    await reachConfirm(user, 'shimmer')
    await user.click(screen.getByRole('button', { name: /start animation/i }))
    await waitFor(() => expect(submitAnimation).toHaveBeenCalledTimes(1))
  })
})

// ── TEST H ───────────────────────────────────────────────────────────────────

describe('TEST H — onAnimateSuccess called after submit', () => {
  it('onAnimateSuccess prop is called exactly once', async () => {
    submitAnimation.mockResolvedValueOnce(mockAnimResult)
    const onAnimateSuccess = vi.fn()
    const user = userEvent.setup()
    render(<AnimateForm onAnimateSuccess={onAnimateSuccess} />)
    await reachConfirm(user, 'shimmer')
    await user.click(screen.getByRole('button', { name: /start animation/i }))
    await waitFor(() => expect(onAnimateSuccess).toHaveBeenCalledTimes(1))
  })
})
