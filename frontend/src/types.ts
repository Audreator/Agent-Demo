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

export interface Citation {
  id: string
  title: string
  url: string
  snippet: string
  reachable: boolean
  status_code: number | null
  checked_at: string | null
  reason?: string
}

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
  citations?: Citation[]
}

export interface ToolCallApi {
  id: string
  session_id: string
  turn_no: number
  tool_name: string
  arguments: Record<string, unknown>
  status: ToolStatus
  retry_count: number
  latency_ms: number
  result_summary: string | null
  result_ref: string | null
  result_meta: { citations?: Citation[]; reachable_count?: number; checked_at?: string } | null
  error: string | null
  created_at: string
}

export interface FeedbackSummary {
  rating: 'like' | 'dislike' | null
  reason_code: string | null
  reason_text: string | null
  likes: number
  dislikes: number
  updated_at: string | null
}

export interface FeedbackResponse {
  message_id: string
  rating: 'like' | 'dislike'
  reason_code: string | null
  reason_text: string | null
  likes: number
  dislikes: number
  updated_at: string
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
  trace_id?: string
  layer?: string
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
  turnTools?: ToolCallApi[]
  citations?: Citation[]
  feedback?: FeedbackSummary | null
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
  feedback?: FeedbackSummary | null
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
  online: boolean
}

export interface ToolResultResponse {
  tool_call_id: string
  content: string
  source: string
  result_ref: string | null
}

export interface TrendPoint {
  bucket: string
  label: string
  turns: number
  tool_success_rate: number | null
  satisfaction_rate: number | null
  p95_ms: number | null
}

export interface AdminAlert {
  rule: string
  name: string
  current: number
  threshold: number
  window: string
  detail: string
  filters: Record<string, unknown>
}

export interface AdminMetrics {
  range: {
    start: string
    end: string
    timezone: string
    granularity: 'hour' | 'day'
  }
  overview: {
    sessions: number
    turns: number
    task_turns: number
    tool_calls: number
    tool_success_rate: number | null
    avg_turn_latency_ms: number | null
    p95_turn_latency_ms: number | null
    total_tokens: number
    likes: number
    dislikes: number
    satisfaction_rate: number | null
    alerts: number
  }
  trend: TrendPoint[]
  tools: Array<{
    tool_name: string
    calls: number
    ok: number
    error: number
    blocked: number
    success_rate: number | null
    avg_ms: number | null
    p95_ms: number | null
    retries: number
    top_errors: Array<{ reason: string; count: number }>
  }>
  models: Array<{
    layer: string
    calls: number
    ok: number
    error: number
    degrade: number
    breaker_open: number
    half_open: number
    probe_success: number
    probe_failure: number
    avg_ms: number | null
    tokens: number
  }>
  context: {
    compress_count: number
    tokens_before_avg: number | null
    tokens_after_avg: number | null
    tokens_before_total: number
    tokens_after_total: number
    budget_wrap_count: number
    max_turn_wrap_count: number
    mode_switch_count: number
  }
  sessions: {
    count: number
    avg_task_turns: number
    distribution: Record<string, number>
    wrapped: number
    native: number
  }
  satisfaction: {
    total: number
    likes: number
    dislikes: number
    rate: number | null
    reasons: Array<{ code: string; count: number }>
    by_tool: Array<{
      tool_name: string
      likes: number
      dislikes: number
      rate: number | null
    }>
    by_model: Array<{
      layer: string
      likes: number
      dislikes: number
      rate: number | null
    }>
  }
  alerts: AdminAlert[]
}

export interface AdminTurn {
  trace_id: string
  session_id: string
  session_title: string
  turn_no: number
  started_at: string
  user_input: string
  final_answer: string
  model_layer: string | null
  finish_reason: string
  tokens: number
  duration_ms: number
  tools: Array<{
    id: string
    name: string
    status: string
    latency_ms: number
    retry_count: number
    error: string | null
    summary: string | null
    result_ref: string | null
    arguments: Record<string, unknown>
    result_meta: Record<string, unknown>
  }>
  feedback: FeedbackSummary | null
}

export interface AdminTurnList {
  total: number
  page: number
  page_size: number
  items: AdminTurn[]
}

export interface TraceEventItem {
  id: number
  trace_id: string
  span_id: string
  parent_span_id: string | null
  kind: string
  payload: Record<string, unknown>
  duration_ms: number | null
  ts: string
}

export interface AdminTurnDetail {
  summary: AdminTurn
  messages: Array<{
    id: string
    role: string
    turn_no: number
    content: string
    meta: MessageMeta
    created_at: string
  }>
  tool_calls: AdminTurn['tools']
  events: TraceEventItem[]
}
