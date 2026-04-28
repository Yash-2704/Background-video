import React, { useState } from 'react'
import PromptForm from './components/PromptForm.jsx'
import AnimateForm from './components/AnimateForm.jsx'
import RunMonitor from './components/RunMonitor.jsx'
import BundleViewer from './components/BundleViewer.jsx'
import PrototypeViewer from './components/PrototypeViewer.jsx'
import './styles/form.css'
import './styles/monitor.css'
import './styles/viewer.css'

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('form')
  const [compileResult, setCompileResult] = useState(null)
  const [formData, setFormData] = useState(null)
  const [runResult, setRunResult] = useState(null)

  function handleCompileSuccess(result, data) {
    setCompileResult(result)
    setFormData(data)
    setCurrentScreen('monitor')
  }

  function handleAnimateSuccess(animResult, data) {
    setCompileResult(animResult)
    setFormData(data)
    setCurrentScreen('monitor')
  }

  function handleMonitorBack() {
    setCompileResult(null)
    setFormData(null)
    setRunResult(null)
    setCurrentScreen('form')
  }

  function handleRunComplete(result) {
    setRunResult(result)
    setCurrentScreen('viewer')
  }

  function handleViewerBack() {
    setCurrentScreen('monitor')
  }

  function headerTitle() {
    if (currentScreen === 'viewer') return 'Background Video — Output Bundle'
    if (currentScreen === 'monitor') return 'Background Video — Run Monitor'
    if (currentScreen === 'animate') return 'Background Video — Animate Image'
    return 'Background Video — Editorial Input'
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>{headerTitle()}</h1>
      </header>
      <main>
        {currentScreen === 'form' && (
          <>
            <PromptForm onCompileSuccess={handleCompileSuccess} />
            <p style={{ textAlign: 'center', marginTop: 12 }}>
              or{' '}
              <button
                className="link-btn"
                onClick={() => setCurrentScreen('animate')}
              >
                animate an image →
              </button>
            </p>
          </>
        )}
        {currentScreen === 'animate' && (
          <>
            <div style={{ maxWidth: 680, margin: '0 auto 0.75rem' }}>
              <button
                className="link-btn"
                onClick={() => setCurrentScreen('form')}
              >
                ← Back to Generate
              </button>
            </div>
            <AnimateForm onAnimateSuccess={handleAnimateSuccess} />
          </>
        )}
        {/* Keep RunMonitor mounted once compileResult is set so its state is
            preserved when BundleViewer is shown and Back is clicked. */}
        {compileResult && (
          <>
            <div style={{ display: currentScreen === 'monitor' ? 'block' : 'none' }}>
              <RunMonitor
                compileResult={compileResult}
                formData={formData}
                onBack={handleMonitorBack}
                onComplete={handleRunComplete}
              />
            </div>
            {currentScreen === 'viewer' && runResult && (
              <BundleViewer
                runResult={runResult}
                onBack={handleViewerBack}
              />
            )}
          </>
        )}
      </main>
    </div>
  )
}
