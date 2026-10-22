export interface SessionInfo {
  id: string
  user_id: string
  title: string
  mode: 'react' | 'native' | string
  task_turn_count: number
  status: string
  last_trace_id: string | null
  created_at: string
  updated_at: string
}

export type ToolStatus = 'running' | 'ok' | 'error' | 'blocked'

export interface ToolCallView {
  id: string
  name: string
  arguments: Record<string, unknown>
  status: ToolStatus
  summary?: string
  error?: string
  latency_ms?: number
  result_ref?: string | null
  attempts?: number
}

export interface MessageMeta {
  kind?: string
  thought?: string
  tool_name?: string
  tool_call_id?: string
  status?: string
  summary?: string
  result_ref?: string | null
  reason?: string
  [key: string]: unknown
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  turn_no: number
  meta: MessageMeta
  created_at?: string
  streaming?: boolean
  thought?: string
  tools?: ToolCallView[]
  contextNote?: string
  finishReason?: string
  errorNote?: string
}

export interface ApiMessage {
  id: string
  session_id: string
  turn_no: number
  role: 'user' | 'assistant' | 'tool'
  content: string
  token_est: number
  meta: MessageMeta
  created_at: string
}

export interface SseFrame {
  event: string
  data: Record<string, unknown>
  id?: string
}

export interface ToolInfo {
  name: string
  description: string
  inputSchema: Record<string, unknown>
}
