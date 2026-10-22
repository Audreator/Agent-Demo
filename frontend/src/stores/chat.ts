import { defineStore } from 'pinia'
import { postMessageStream } from '@/api/sse'
import type { ApiMessage, ChatMessage, SessionInfo, SseFrame, ToolCallView } from '@/types'

interface StreamState {
  aborted: boolean
  done: boolean
}

const newId = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `local-${Date.now()}-${Math.random().toString(16).slice(2)}`

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `请求失败 (${response.status})`)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function toDisplayMessage(message: ApiMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    turn_no: message.turn_no,
    meta: message.meta ?? {},
    created_at: message.created_at,
    thought: typeof message.meta?.thought === 'string' ? message.meta.thought : undefined,
    errorNote:
      message.meta?.kind === 'parse_error' && typeof message.meta?.reason === 'string'
        ? message.meta.reason
        : undefined,
  }
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: [] as SessionInfo[],
    currentSessionId: null as string | null,
    messages: [] as ChatMessage[],
    sending: false,
    loadingMessages: false,
    connection: 'idle' as 'idle' | 'streaming' | 'reconnecting' | 'error',
    error: '',
    lastSeq: 0,
    activeTraceId: '',
    abortController: null as AbortController | null,
    streamState: { aborted: false, done: false } as StreamState,
    messageLoadToken: 0,
  }),
  getters: {
    currentSession(state): SessionInfo | null {
      return state.sessions.find((item) => item.id === state.currentSessionId) ?? null
    },
  },
  actions: {
    async loadSessions() {
      const sessions = await api<SessionInfo[]>('/api/sessions')
      this.sessions = sessions
      if (!this.currentSessionId && sessions.length) {
        await this.selectSession(sessions[0].id)
      }
    },

    async createSession() {
      const session = await api<SessionInfo>('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: '新会话' }),
      })
      this.sessions.unshift(session)
      await this.selectSession(session.id)
      return session
    },

    async selectSession(sessionId: string) {
      if (this.sending) {
        return
      }
      this.currentSessionId = sessionId
      this.error = ''
      this.lastSeq = 0
      this.activeTraceId = ''
      this.messageLoadToken += 1
      await this.loadMessages(sessionId, this.messageLoadToken)
    },

    async loadMessages(sessionId: string, token?: number) {
      this.loadingMessages = true
      try {
        const messages = await api<ApiMessage[]>(`/api/sessions/${sessionId}/messages`)
        if (
          this.currentSessionId === sessionId &&
          (token === undefined || token === this.messageLoadToken)
        ) {
          this.messages = messages.map(toDisplayMessage)
        }
      } finally {
        if (token === undefined || token === this.messageLoadToken) {
          this.loadingMessages = false
        }
      }
    },

    async deleteSession(sessionId: string) {
      if (this.sending) {
        return
      }
      await api<void>(`/api/sessions/${sessionId}`, { method: 'DELETE' })
      this.sessions = this.sessions.filter((item) => item.id !== sessionId)
      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null
        this.messages = []
        if (this.sessions.length) {
          await this.selectSession(this.sessions[0].id)
        }
      }
    },

    async send(content: string) {
      const text = content.trim()
      if (!text || this.sending || !this.currentSessionId || this.loadingMessages) {
        return
      }
      const sessionId = this.currentSessionId
      const clientMessageId = newId()
      const turnNo = this.messages.reduce((max, item) => Math.max(max, item.turn_no), 0) + 1
      this.error = ''
      this.lastSeq = 0
      this.activeTraceId = ''
      this.sending = true
      this.connection = 'streaming'
      const controller = new AbortController()
      this.abortController = controller
      this.streamState = { aborted: false, done: false }

      this.messages.push({
        id: newId(),
        role: 'user',
        content: text,
        turn_no: turnNo,
        meta: {},
      })
      const assistant: ChatMessage = {
        id: newId(),
        role: 'assistant',
        content: '',
        turn_no: turnNo,
        meta: {},
        streaming: true,
        thought: '',
        tools: [],
      }
      this.messages.push(assistant)
      const liveAssistant = this.messages[this.messages.length - 1]

      try {
        let attempt = 0
        let receivedAny = false
        while (!this.streamState.done && attempt < 3) {
          try {
            await postMessageStream(sessionId, text, {
              lastEventId: receivedAny && this.lastSeq > 0 ? this.lastSeq : null,
              clientMessageId,
              signal: controller.signal,
              onFrame: (frame) => {
                receivedAny = true
                this.handleFrame(frame, liveAssistant)
              },
            })
          } catch (err) {
            if (this.streamState.aborted || controller.signal.aborted) {
              break
            }
            attempt += 1
            if (attempt >= 3) {
              throw err
            }
            this.connection = 'reconnecting'
            await new Promise((resolve) => setTimeout(resolve, 400 * attempt))
            continue
          }
          if (!this.streamState.done && !this.streamState.aborted) {
            attempt += 1
            this.connection = 'reconnecting'
          }
        }
        if (!this.streamState.done && !this.streamState.aborted) {
          this.error = '连接中断，回复可能不完整，请重试。'
          this.connection = 'error'
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          this.error = err instanceof Error ? err.message : String(err)
          this.connection = 'error'
        }
      } finally {
        liveAssistant.streaming = false
        this.sending = false
        this.abortController = null
        if (this.connection !== 'error') {
          this.connection = 'idle'
        }
        if (!controller.signal.aborted && (!this.streamState.done || this.error)) {
          try {
            await this.loadMessages(sessionId, this.messageLoadToken)
          } catch {
            // 失败时以服务端已落库记录为准，避免残留未发送的乐观消息
          }
        }
        try {
          const sessions = await api<SessionInfo[]>('/api/sessions')
          this.sessions = sessions
        } catch {
          // keep current session list when refresh fails
        }
      }
    },

    stop() {
      if (!this.abortController) {
        return
      }
      this.streamState.aborted = true
      this.abortController.abort()
      this.sending = false
      this.connection = 'idle'
      const last = this.messages[this.messages.length - 1]
      if (last?.role === 'assistant') {
        last.streaming = false
      }
    },

    handleFrame(frame: SseFrame, assistant: ChatMessage) {
      const data = frame.data
      if (typeof data.seq === 'number') {
        this.lastSeq = data.seq
      }
      const turnNo = typeof data.turn_no === 'number' ? data.turn_no : assistant.turn_no
      assistant.turn_no = turnNo

      switch (frame.event) {
        case 'meta': {
          this.activeTraceId = String(data.trace_id ?? '')
          assistant.meta.trace_id = this.activeTraceId
          if (typeof data.mode === 'string') {
            assistant.meta.mode = data.mode
          }
          break
        }
        case 'thought_delta': {
          assistant.thought = (assistant.thought ?? '') + String(data.delta ?? '')
          break
        }
        case 'message_delta': {
          assistant.content += String(data.delta ?? '')
          break
        }
        case 'tool_start': {
          const tool: ToolCallView = {
            id: String(data.tool_call_id ?? newId()),
            name: String(data.name ?? 'tool'),
            arguments: (data.arguments as Record<string, unknown>) ?? {},
            status: 'running',
          }
          assistant.tools = [...(assistant.tools ?? []), tool]
          break
        }
        case 'tool_result': {
          const tools = assistant.tools ?? []
          const target = tools.find((item) => item.id === String(data.tool_call_id ?? ''))
          if (target) {
            target.status = data.status === 'ok' ? 'ok' : data.status === 'blocked' ? 'blocked' : 'error'
            target.summary = typeof data.summary === 'string' ? data.summary : undefined
            target.error = typeof data.error === 'string' ? data.error : undefined
            target.latency_ms =
              typeof data.latency_ms === 'number' ? data.latency_ms : undefined
            target.result_ref =
              typeof data.result_ref === 'string' ? data.result_ref : target.result_ref
            target.attempts = typeof data.attempts === 'number' ? data.attempts : target.attempts
          }
          break
        }
        case 'context_update': {
          if (data.reset) {
            assistant.content = ''
            assistant.thought = ''
            assistant.contextNote = undefined
            assistant.errorNote = undefined
            break
          }
          const parts: string[] = []
          if (data.mode) {
            parts.push(`模式切换为 ${String(data.mode)}`)
          }
          if (data.compressed) {
            parts.push(
              `上下文已压缩 ${String(data.tokens_before ?? '?')} → ${String(
                data.tokens_after ?? '?',
              )} tokens`,
            )
          }
          assistant.contextNote = parts.join('；')
          break
        }
        case 'error': {
          const message = String(data.message ?? '发生错误')
          this.error = message
          this.connection = 'error'
          break
        }
        case 'done': {
          this.streamState.done = true
          assistant.finishReason = String(data.finish_reason ?? '')
          assistant.streaming = false
          break
        }
        default:
          break
      }
    },
  },
})
