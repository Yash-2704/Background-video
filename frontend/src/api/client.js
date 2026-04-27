const API_BASE = 'http://100.92.126.27:8000'
export async function parsePrompt(prompt) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/v1/parse-prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    })
  } catch (networkError) {
    throw new Error(`Network error reaching parse endpoint: ${networkError.message}`)
  }

  if (!response.ok) {
    if (response.status === 500) {
      throw new Error('The interpret service is not available yet. Please contact the operator.')
    }
    const text = await response.text()
    throw new Error(`Parse failed: ${response.status} ${text}`)
  }

  return response.json()
}

export async function compilePrompts(formData) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/v1/compile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    })
  } catch (networkError) {
    throw new Error(`Network error reaching compile endpoint: ${networkError.message}`)
  }

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Compile failed: ${response.status} ${text}`)
  }

  return response.json()
}

export async function submitGeneration(runPayload) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/v1/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(runPayload),
    })
  } catch (networkError) {
    throw new Error(`Network error reaching generate endpoint: ${networkError.message}`)
  }

  // 409 means the run is already in progress (React StrictMode fires useEffect
  // twice in dev — the second POST arrives after the first fire-and-forget POST
  // already registered the run). Treat it as success and start polling.
  if (response.status === 409) {
    return { run_id: runPayload.run_id, status: 'running' }
  }

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Generation failed: ${response.status} ${text}`)
  }

  return response.json()
}

export async function getRunStatus(run_id) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/v1/run/${run_id}/status`)
  } catch (networkError) {
    throw new Error(`Network error reaching status endpoint: ${networkError.message}`)
  }

  if (response.status === 404) {
    return { status: 'pending', stages: {} }
  }

  if (!response.ok) {
    throw new Error(`Status check failed: ${response.status}`)
  }

  return response.json()
}

export async function fetchBundleFile(clip_id, filename) {
  let response
  try {
    response = await fetch(`${API_BASE}/api/v1/bundle/${clip_id}/${filename}`)
  } catch (networkError) {
    throw new Error(`Network error reaching bundle endpoint: ${networkError.message}`)
  }

  if (response.status === 404) {
    return null
  }

  if (!response.ok) {
    throw new Error(`Bundle fetch failed: ${response.status}`)
  }

  return response.json()
}

export function getMediaUrl(clip_id, filename) {
  return `${API_BASE}/api/v1/media/${clip_id}/${filename}`
}
