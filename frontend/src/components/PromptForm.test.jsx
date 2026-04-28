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

// ── TEST A — renders textarea on initial load ─────────────────────────────────

describe('TEST A — renders textarea on initial load', () => {
  it('textarea element is present', () => {
    render(<PromptForm />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })
})

// ── TEST B — single step render ───────────────────────────────────────────────

describe('TEST B — single step render', () => {
  it('no Interpret button, Generate button present, textarea present', () => {
    render(<PromptForm />)
    expect(screen.queryByRole('button', { name: /interpret/i })).toBeNull()
    expect(screen.getByRole('button', { name: /generate/i })).toBeInTheDocument()
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })
})

// ── TEST C — Generate button disabled when textarea is empty ──────────────────

describe('TEST C — Generate button disabled when textarea is empty', () => {
  it('button is disabled initially', () => {
    render(<PromptForm />)
    expect(screen.getByRole('button', { name: /generate/i })).toBeDisabled()
  })
})

// ── TEST D — Generate button enabled after typing ─────────────────────────────

describe('TEST D — Generate button enabled after typing', () => {
  it('button becomes enabled when text is entered', async () => {
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'a')
    expect(screen.getByRole('button', { name: /generate/i })).toBeEnabled()
  })
})

// ── TEST E — Generate button calls compilePrompts directly ────────────────────

describe('TEST E — Generate button calls compilePrompts directly', () => {
  it('compilePrompts called with prompt string, parsePrompt never called', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'rainy night city')
    await user.click(screen.getByRole('button', { name: /generate/i }))
    await waitFor(() => expect(compilePrompts).toHaveBeenCalledWith('rainy night city'))
    expect(parsePrompt).not.toHaveBeenCalled()
  })
})

// ── TEST F — onCompileSuccess called after successful compile ─────────────────

describe('TEST F — onCompileSuccess called after compile', () => {
  it('onCompileSuccess is called exactly once with result and raw_prompt', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const onCompileSuccess = vi.fn()
    const user = userEvent.setup()
    render(<PromptForm onCompileSuccess={onCompileSuccess} />)
    await user.type(screen.getByRole('textbox'), 'night city plaza')
    await user.click(screen.getByRole('button', { name: /generate/i }))
    await waitFor(() => expect(onCompileSuccess).toHaveBeenCalledTimes(1))
    const [result, meta] = onCompileSuccess.mock.calls[0]
    expect(result).toEqual(mockCompileResult)
    expect(meta).toEqual({ raw_prompt: 'night city plaza' })
  })
})

// ── TEST G — error from compilePrompts is displayed ───────────────────────────

describe('TEST G — error from compilePrompts is displayed', () => {
  it('error message is visible when compilePrompts rejects', async () => {
    compilePrompts.mockRejectedValueOnce(new Error('Compile failed: 500'))
    const user = userEvent.setup()
    render(<PromptForm />)
    await user.type(screen.getByRole('textbox'), 'any prompt')
    await user.click(screen.getByRole('button', { name: /generate/i }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(screen.getByRole('alert').textContent).toMatch(/compile failed/i)
  })
})
