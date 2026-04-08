import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent } from '@testing-library/react'
import EditorialForm from './EditorialForm.jsx'
import FORM_OPTIONS from '../config/formOptions.js'

vi.mock('../api/client', () => ({
  compilePrompts: vi.fn(),
}))

import { compilePrompts } from '../api/client'

const mockCompileResult = {
  input_hash_short:  'b2e7f3',
  selected_lut:      'cool_authority',
  lower_third_style: 'minimal_dark_bar',
  positive_hash:     'b2e7f3a1c9d4e8f2',
  motion_hash:       '9fa14c82...',
  negative_hash:     'c4d8b901...',
}

// Helper: select a value in every one of the 6 fields
async function fillAllFields(user) {
  const selects = screen.getAllByRole('combobox')
  const fields = ['category', 'location_feel', 'time_of_day', 'color_temperature', 'mood', 'motion_intensity']
  for (let i = 0; i < fields.length; i++) {
    const firstOpt = FORM_OPTIONS[fields[i]][0]
    await user.selectOptions(selects[i], firstOpt.value)
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

// ── Rendering ───────────────────────────────────────────────────────────────

describe('Rendering', () => {
  it('1. renders without crashing', () => {
    render(<EditorialForm />)
    expect(document.querySelector('form')).toBeTruthy()
  })

  it('2. all 6 field labels are present', () => {
    render(<EditorialForm />)
    expect(screen.getByText('Category')).toBeInTheDocument()
    expect(screen.getByText('Location Feel')).toBeInTheDocument()
    expect(screen.getByText('Time of Day')).toBeInTheDocument()
    expect(screen.getByText('Color Temperature')).toBeInTheDocument()
    expect(screen.getByText('Mood')).toBeInTheDocument()
    expect(screen.getByText('Motion Intensity')).toBeInTheDocument()
  })

  it('3. all 6 select elements are rendered', () => {
    render(<EditorialForm />)
    expect(screen.getAllByRole('combobox')).toHaveLength(6)
  })

  it('4. submit button has text "Compile Prompts"', () => {
    render(<EditorialForm />)
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
  })

  it('5. submit button is disabled on initial render', () => {
    render(<EditorialForm />)
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeDisabled()
  })

  it('6. no error message visible on initial render', () => {
    render(<EditorialForm />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('7. no result panel visible on initial render', () => {
    render(<EditorialForm />)
    expect(screen.queryByText('Compile Result')).not.toBeInTheDocument()
  })
})

// ── Validation ──────────────────────────────────────────────────────────────

describe('Validation', () => {
  it('8. submit button remains disabled when only 5 of 6 fields are selected', async () => {
    const user = userEvent.setup()
    render(<EditorialForm />)
    const selects = screen.getAllByRole('combobox')
    // Fill only first 5
    const fields = ['category', 'location_feel', 'time_of_day', 'color_temperature', 'mood']
    for (let i = 0; i < 5; i++) {
      await user.selectOptions(selects[i], FORM_OPTIONS[fields[i]][0].value)
    }
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeDisabled()
  })

  it('9. submit button becomes enabled when all 6 fields have a value', async () => {
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeEnabled()
  })

  it('10. clearing one field after filling all 6 disables submit again', async () => {
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    const selects = screen.getAllByRole('combobox')
    // Clear the first select back to empty — userEvent cannot pick disabled placeholders
    fireEvent.change(selects[0], { target: { value: '' } })
    expect(screen.getByRole('button', { name: /compile prompts/i })).toBeDisabled()
  })
})

// ── Submission — success path ────────────────────────────────────────────────

describe('Submission — success path', () => {
  it('11. selecting all 6 fields and clicking submit calls compilePrompts exactly once', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await waitFor(() => expect(compilePrompts).toHaveBeenCalledTimes(1))
  })

  it('12. compilePrompts is called with all 6 field keys non-empty', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await waitFor(() => expect(compilePrompts).toHaveBeenCalled())
    const payload = compilePrompts.mock.calls[0][0]
    expect(Object.keys(payload)).toEqual(
      expect.arrayContaining([
        'category', 'location_feel', 'time_of_day',
        'color_temperature', 'mood', 'motion_intensity',
      ])
    )
    Object.values(payload).forEach((v) => expect(v).not.toBe(''))
  })

  it('13. "Compiling..." appears in the button during the pending state', async () => {
    let resolve
    compilePrompts.mockReturnValueOnce(new Promise((r) => { resolve = r }))
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByRole('button', { name: /compiling/i })).toBeInTheDocument()
    // Resolve the promise and flush remaining state updates before the test ends
    await waitFor(() => { resolve(mockCompileResult) })
    await screen.findByRole('button', { name: /compile prompts/i })
  })

  it('14. result panel appears after successful compile showing input_hash_short', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByText('b2e7f3')).toBeInTheDocument()
  })

  it('15. result panel shows selected_lut', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByText('cool_authority')).toBeInTheDocument()
  })

  it('16. result panel shows lower_third_style', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByText('minimal_dark_bar')).toBeInTheDocument()
  })

  it('17. "Proceed to Generation" button is present in result panel and is disabled', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    const proceedBtn = await screen.findByRole('button', { name: /proceed to generation/i })
    expect(proceedBtn).toBeDisabled()
  })

  it('18. submit button returns to "Compile Prompts" after response received', async () => {
    compilePrompts.mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /compile prompts/i })).toBeInTheDocument()
    )
  })
})

// ── Submission — error path ──────────────────────────────────────────────────

describe('Submission — error path', () => {
  it('19. when compilePrompts rejects, error message appears', async () => {
    compilePrompts.mockRejectedValueOnce(new Error('Compile failed: 500 Server Error'))
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByText(/compile failed/i)).toBeInTheDocument()
  })

  it('20. error element has role="alert"', async () => {
    compilePrompts.mockRejectedValueOnce(new Error('Compile failed: 500'))
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
  })

  it('21. isLoading returns to false after error (submit re-enables)', async () => {
    compilePrompts.mockRejectedValueOnce(new Error('Error'))
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /compile prompts/i })).toBeEnabled()
    )
  })

  it('22. submitting again after an error clears the previous error', async () => {
    compilePrompts
      .mockRejectedValueOnce(new Error('First error'))
      .mockResolvedValueOnce(mockCompileResult)
    const user = userEvent.setup()
    render(<EditorialForm />)
    await fillAllFields(user)
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await screen.findByRole('alert')
    await user.click(screen.getByRole('button', { name: /compile prompts/i }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  })
})

// ── Option set integrity ─────────────────────────────────────────────────────

describe('Option set integrity', () => {
  it('23. category select has exactly 7 non-placeholder options', () => {
    render(<EditorialForm />)
    const selects = screen.getAllByRole('combobox')
    const categorySelect = selects[0]
    const nonPlaceholder = Array.from(categorySelect.options).filter((o) => o.value !== '')
    expect(nonPlaceholder).toHaveLength(7)
  })

  it('24. color_temperature select has exactly 3 options', () => {
    render(<EditorialForm />)
    const selects = screen.getAllByRole('combobox')
    const ctSelect = selects[3]
    const nonPlaceholder = Array.from(ctSelect.options).filter((o) => o.value !== '')
    expect(nonPlaceholder).toHaveLength(3)
  })

  it('25. motion_intensity select has exactly 3 options', () => {
    render(<EditorialForm />)
    const selects = screen.getAllByRole('combobox')
    const miSelect = selects[5]
    const nonPlaceholder = Array.from(miSelect.options).filter((o) => o.value !== '')
    expect(nonPlaceholder).toHaveLength(3)
  })

  it('26. time_of_day select has exactly 4 options (Day, Dusk, Night, N/A)', () => {
    render(<EditorialForm />)
    const selects = screen.getAllByRole('combobox')
    const todSelect = selects[2]
    const nonPlaceholder = Array.from(todSelect.options).filter((o) => o.value !== '')
    expect(nonPlaceholder).toHaveLength(4)
    const values = nonPlaceholder.map((o) => o.value)
    expect(values).toEqual(['Day', 'Dusk', 'Night', 'N/A'])
  })
})

// ── formOptions config ───────────────────────────────────────────────────────

describe('formOptions config', () => {
  it('27. FORM_OPTIONS has exactly 6 keys', () => {
    expect(Object.keys(FORM_OPTIONS)).toHaveLength(6)
  })

  it('28. every option object has value, label, helper fields', () => {
    for (const options of Object.values(FORM_OPTIONS)) {
      for (const opt of options) {
        expect(opt).toHaveProperty('value')
        expect(opt).toHaveProperty('label')
        expect(opt).toHaveProperty('helper')
      }
    }
  })

  it('29. color_temperature values are exactly ["Neutral", "Cool", "Warm"]', () => {
    const values = FORM_OPTIONS.color_temperature.map((o) => o.value)
    expect(values).toEqual(['Neutral', 'Cool', 'Warm'])
  })
})
