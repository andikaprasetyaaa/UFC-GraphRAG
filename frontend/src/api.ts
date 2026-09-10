export type AskResponse = {
  answer: string
  sources: Array<{ doc_type?: string; ref?: string | number; preview?: string }>
  cache_hit?: string | boolean
  structured_facts_used?: boolean
  elapsed_seconds?: number
}

export async function askRag(question: string): Promise<AskResponse> {
  const res = await fetch('http://localhost:8000/api/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const data = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(data.detail || 'Request failed')
  }

  return res.json()
}
