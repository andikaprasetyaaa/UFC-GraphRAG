import React, { useMemo, useState } from 'react'
import ReactDOM from 'react-dom/client'
import './styles.css'
import { askRag } from './api'

function renderAnswer(text: string) {
  const cleanText = text.replace(/\s*\[\d+(?:\s*,\s*\d+)*\]/g, '')

  return cleanText.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>
    }

    return <React.Fragment key={index}>{part}</React.Fragment>
  })
}

function App() {
  const [question, setQuestion] = useState('')
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const quickQuestions = useMemo(
    () => [
      'Siapa yang mengalahkan Ilia Topuria?',
      'Bagaimana rekor head to head Jon Jones vs Daniel Cormier?',
      'Siapa itu Khabib Nurmagomedov?',
    ],
    [],
  )

  const handleAsk = async (value?: string) => {
    const text = (value ?? question).trim()
    if (!text) return

    setQuestion('')
    setSubmittedQuestion(text)
    setLoading(true)
    setError('')
    setAnswer('')

    try {
      const res = await askRag(text)
      setAnswer(res.answer || 'Tidak ada jawaban yang bisa diambil dari konteks.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Gagal menghubungi backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-shell">
      <div className="bg-grid" />
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />

      <main className="chat-panel">
        <div className="live-video-layer" aria-hidden="true">
          <div className="signal-ring signal-one" />
          <div className="signal-ring signal-two" />
          <div className="signal-glow" />
        </div>

        <header className="chat-header">
          <div>
            <p className="eyebrow">UFC intelligence</p>
            <h1>Ask the fight AI.</h1>
          </div>

          <div className="status-pill">
            <span className="pulse-dot" />
            Live
          </div>
        </header>

        <div className="quick-row">
          {quickQuestions.map((q) => (
            <button key={q} className="chip" onClick={() => handleAsk(q)}>
              {q}
            </button>
          ))}
        </div>

        <div className="messages">
          {submittedQuestion && (
            <div className="message user">
              <div className="avatar-user">U</div>
              <div className="bubble">{submittedQuestion}</div>
            </div>
          )}

          <div className="message bot">
            <div className="avatar-bot">AI</div>
            <div className="bubble bot-bubble">
              {loading
                ? 'Analyzing the fight database...'
                : error
                  ? error
                  : answer
                    ? renderAnswer(answer)
                    : 'Ask me anything about fighters, records, and matchup history. I will answer with context from the UFC dataset.'}
            </div>
          </div>
        </div>

        <div className="composer">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAsk()
            }}
            placeholder="Type your question..."
          />
          <button className="send-btn" onClick={() => handleAsk()} disabled={loading}>
            {loading ? '...' : 'Send'}
          </button>
        </div>
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
