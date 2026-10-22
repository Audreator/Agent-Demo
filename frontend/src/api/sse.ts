import type { SseFrame } from '@/types'

export interface StreamOptions {
  signal?: AbortSignal
  lastEventId?: number | null
  clientMessageId?: string
  onFrame: (frame: SseFrame) => void
}

export async function postMessageStream(
  sessionId: string,
  content: string,
  options: StreamOptions,
): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      content,
      last_event_id: options.lastEventId ?? null,
      client_message_id: options.clientMessageId ?? null,
    }),
    signal: options.signal,
  })

  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown }
      if (typeof parsed.detail === 'string') {
        detail = parsed.detail
      }
    } catch {
      // keep raw text
    }
    throw new Error(`请求失败 (${response.status}): ${detail}`)
  }
  if (!response.body) {
    throw new Error('浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      buffer += decoder.decode()
      break
    }
    buffer += decoder.decode(value, { stream: true })
    buffer = consumeFrames(buffer, options.onFrame)
  }
  if (buffer.trim()) {
    consumeFrames(buffer + '\n\n', options.onFrame)
  }
}

function consumeFrames(buffer: string, onFrame: (frame: SseFrame) => void): string {
  let normalized = buffer.replace(/\r\n/g, '\n')
  let separator = normalized.indexOf('\n\n')
  while (separator !== -1) {
    const raw = normalized.slice(0, separator)
    normalized = normalized.slice(separator + 2)
    const frame = parseFrame(raw)
    if (frame) {
      onFrame(frame)
    }
    separator = normalized.indexOf('\n\n')
  }
  return normalized
}

function parseFrame(raw: string): SseFrame | null {
  let event = 'message'
  let id: string | undefined
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (!line || line.startsWith(':')) {
      continue
    }
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) {
      value = value.slice(1)
    }
    if (field === 'event') {
      event = value
    } else if (field === 'id') {
      id = value
    } else if (field === 'data') {
      dataLines.push(value)
    }
  }
  if (!dataLines.length) {
    return null
  }
  try {
    const data = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
    return { event, data, id }
  } catch {
    return null
  }
}
