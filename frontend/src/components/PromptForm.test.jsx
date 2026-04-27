import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import PromptForm from './PromptForm.jsx'

vi.mock('../api/client', () => ({
  parsePrompt:    vi.fn(),
  compilePrompts: vi.fn(),
}))

import { parsePrompt, compilePrompts } from '../api/client'

const mockParsed = {
  category:          'Government',
  location_feel:     'Urban',
  time_of_day:       'Night',
  color_temperature: 'Cool',
  mood:              'Tense',
  motion_intensity:  'slow',
  original_prompt:   'rainy night city',
  inference_notes:   'Inferred tense mood from "government plaza" phrasing.',
}

const mockCompileResult = {
  input_hash_short:  'abc123',
  selected_lut:      'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  positive_hash:     'abc123def456abcd',
  motion_hash:       '9fa14c82',
  negative_hash:     'c4d8b901',
}

beforeEach(() => {
  vi.clearAllMocks()
})

// ── Helper: advance through both steps ──────────────────────────────────────

async function reachConfirm(user, promptText = 'rainy night city') {
  parsePrompt.mockResolvedValueOnce(mockParsed)
  await user.type(screen.getByRole('textbox'), promptText)
  await user.click(screen.getByRole('button', { name: /interpret/i }))
  await screen.findByText('Interpreted as:')
}

// ── TEST A ───────────────────────────────────────────────────────────────────

describe('TEST A — renders textarea on initial load', () => {
  it('textarea element is present', () => {
    render(<PromptForm />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })
})

// ── TEST B ───────────────────────────────────────────────────────────────────

describe('TEST B — Interpret button disabled when textarea is empty', () => {
  it('button is disabled initially', () => {
    render(<PromptForm />)
    expect(screen.getByRole('button', { name: /interpret/i })).toBeDisabled()
  })
})

// ── TEST C ───────────────────────────────────────────────────────────────────

describe('TEST C — Interpret button enabled after typing', () => {
  it('button becomes enabled when text is entered', async () => {
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'a')
    expect(screen.getByRole('button', { name: /interpret/i })).toBeEnabled()
  })
})

// ── TEST D ───────────────────────────────────────────────────────────────────

describe('TEST D — clicking Interpret calls parsePrompt with the typed text', () => {
  it('parsePrompt receives the exact prompt text', async () => {
    parsePrompt.mockResolvedValueOnce(mockParsed)
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'rainy night city')
    await user.click(screen.getByRole('button', { name: /interpret/i }))
    await waitFor(() => expect(parsePrompt).toHaveBeenCalledWith('rainy night city'))
  })
})

// ── TEST E ───────────────────────────────────────────────────────────────────

describe('TEST E — after successful parse, interpretation table appears', () => {
  it('heading and at least one inferred field value are visible', async () => {
    const user = userEvent.setup()
    render(<PromptForm />)
    await reachConfirm(user)
    expect(screen.getByText('Interpreted as:')).toBeInTheDocument()
    expect(screen.getByText('Government')).toBeInTheDocument()
  })
})

// ── TEST F ───────────────────────────────────────────────────────────────────

describe('TEST F — Edit button returns to textarea with prompt preserved', () => {
  it('textarea is shown again with original text after clicking Edit', async () => {
    const user = userEvent.setup()
    render(<PromptForm />)
    await reachConfirm(user, 'rainy night city')
    await user.click(screen.getByRole('button', { name: /edit/i }))
    const textarea = screen.getByRole('textbox')
    expect(textarea).toBeInTheDocument()
    expect(textarea.value).toBe('rainy night city')
  })
})

// ── TEST G ───────────────────────────────────────────────────────────────────

describe('TEST G — Confirm button calls compilePrompts with 6 inferred fields', () => {
  it('compilePrompts is called without original_prompt or inference_notes', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<PromptForm />)
    await reachConfirm(user)
    await user.click(screen.getByRole('button', { name: /confirm/i }))
    await waitFor(() => expect(compilePrompts).toHaveBeenCalledTimes(1))
    const payload = compilePrompts.mock.calls[0][0]
    expect(Object.keys(payload)).toEqual(
      expect.arrayContaining([
        'category', 'location_feel', 'time_of_day',
        'color_temperature', 'mood', 'motion_intensity',
      ])
    )
    expect(payload).not.toHaveProperty('original_prompt')
    expect(payload).not.toHaveProperty('inference_notes')
  })
})

// ── TEST H ───────────────────────────────────────────────────────────────────

describe('TEST H — 500 from parsePrompt shows operator message', () => {
  it('operator contact message is displayed', async () => {
    parsePrompt.mockRejectedValueOnce(new Error('500'))
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'any prompt')
    await user.click(screen.getByRole('button', { name: /interpret/i }))
    expect(await screen.findByText(/interpret service is not available/i)).toBeInTheDocument()
  })
})

// ── TEST I ───────────────────────────────────────────────────────────────────

describe('TEST I — onCompileSuccess called after successful confirm', () => {
  it('onCompileSuccess is called exactly once', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const onCompileSuccess = vi.fn()
    const user = userEvent.setup()
    render(<PromptForm onCompileSuccess={onCompileSuccess} />)
    await reachConfirm(user)
    await user.click(screen.getByRole('button', { name: /confirm/i }))
    await waitFor(() => expect(onCompileSuccess).toHaveBeenCalledTimes(1))
  })
})
