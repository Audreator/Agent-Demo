<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import AdminTrend from '@/components/AdminTrend.vue'
import { api, fmtDateTime, fmtMs, fmtPercent, loadAppConfig } from '@/api/client'
import type { AdminMetrics, AdminTurnList } from '@/types'

const router = useRouter()

function localInput(date: Date): string {
  const parts = new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date)
  return parts.replace(' ', 'T')
}

const now = new Date(Date.now() + 60_000)
const start = ref(localInput(new Date(now.getTime() - 7 * 24 * 3600 * 1000)))
const end = ref(localInput(now))
const metrics = ref<AdminMetrics | null>(null)
const turns = ref<AdminTurnList | null>(null)
const loading = ref(false)
const exporting = ref(false)
const error = ref('')
const page = ref(1)
const pageSize = ref(50)
const tab = ref<'tools' | 'models' | 'context' | 'sessions' | 'satisfaction'>('tools')
const turnsSection = ref<HTMLElement | null>(null)

const REASON_LABELS: Record<string, string> = {
  irrelevant: '答非所问',
  tool_error: '工具调用错误',
  bad_link: '链接失效',
  too_slow: '响应太慢',
  fact_error: '事实错误',
  incomplete: '内容不完整',
  other: '其他',
}

const filters = reactive({
  session_id: '',
  tool_name: '',
  tool_status: '',
  rating: '',
  finish_reason: '',
  trace_id: '',
})

const toolOptions = ['calculator', 'todo', 'search', 'weather', 'read_docs']
const finishOptions = [
  { value: 'final_answer', label: '正常完成' },
  { value: 'wrap_up', label: '收束总结' },
  { value: 'wrap_up:budget_exceeded', label: '预算收束' },
  { value: 'wrap_up:max_task_turns', label: '30 轮触顶' },
  { value: 'wrap_up:max_model_calls', label: '模型调用次数触顶' },
  { value: 'parse_error', label: '解析错误' },
  { value: 'error', label: '模型/服务错误' },
  { value: 'budget_exceeded', label: '预算超限' },
]

function toIso(value: string): string {
  if (!value) {
    return ''
  }
  if (/[zZ]|[+-]\d{2}:\d{2}$/.test(value)) {
    return value
  }
  return `${value.length === 16 ? `${value}:00` : value}+08:00`
}

function queryParams(withPaging: boolean): string {
  const params = new URLSearchParams({ start: toIso(start.value), end: toIso(end.value) })
  for (const [key, value] of Object.entries(filters)) {
    if (value) {
      params.set(key, value)
    }
  }
  if (withPaging) {
    params.set('page', String(page.value))
    params.set('page_size', String(pageSize.value))
  }
  return params.toString()
}

async function loadAll(): Promise<void> {
  if (!start.value || !end.value) {
    error.value = '请选择完整的时间范围'
    return
  }
  loading.value = true
  error.value = ''
  try {
    const [metricsResponse, turnsResponse] = await Promise.all([
      api<AdminMetrics>(`/api/admin/metrics?${queryParams(false)}`),
      api<AdminTurnList>(`/api/admin/turns?${queryParams(true)}`),
    ])
    metrics.value = metricsResponse
    turns.value = turnsResponse
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function applyFilters(): void {
  page.value = 1
  void loadAll()
}

function resetFilters(): void {
  for (const key of Object.keys(filters) as Array<keyof typeof filters>) {
    filters[key] = ''
  }
  page.value = 1
  void loadAll()
}

async function applyAlert(alertFilters: Record<string, unknown>): Promise<void> {
  const current = new Date(Date.now() + 60_000)
  start.value = localInput(new Date(current.getTime() - 3600 * 1000))
  end.value = localInput(current)
  resetFiltersWithoutLoad()
  if (typeof alertFilters.tool_status === 'string') {
    filters.tool_status = alertFilters.tool_status
  }
  if (typeof alertFilters.rating === 'string') {
    filters.rating = alertFilters.rating
  }
  if (typeof alertFilters.finish_reason === 'string') {
    filters.finish_reason = alertFilters.finish_reason
  }
  page.value = 1
  await loadAll()
  await nextTick()
  turnsSection.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function resetFiltersWithoutLoad(): void {
  for (const key of Object.keys(filters) as Array<keyof typeof filters>) {
    filters[key] = ''
  }
}

function quickRange(hours: number): void {
  const current = new Date(Date.now() + 60_000)
  start.value = localInput(new Date(current.getTime() - hours * 3600 * 1000))
  end.value = localInput(current)
  page.value = 1
  void loadAll()
}

function filterByTool(name: string): void {
  filters.tool_name = name
  tab.value = 'tools'
  page.value = 1
  void loadAll()
}

function openTurn(traceId: string): void {
  void router.push(`/admin/turns/${traceId}`)
}

async function exportCsv(): Promise<void> {
  if (exporting.value) {
    return
  }
  exporting.value = true
  error.value = ''
  try {
    const response = await fetch(`/api/admin/export?${queryParams(false)}`)
    if (!response.ok) {
      let detail = `导出失败 (${response.status})`
      try {
        const payload = (await response.json()) as { detail?: unknown }
        if (typeof payload.detail === 'string') {
          detail = payload.detail
        }
      } catch {
        // keep default detail
      }
      throw new Error(detail)
    }
    const blob = await response.blob()
    const disposition = response.headers.get('content-disposition') ?? ''
    const matched = /filename="([^"]+)"/.exec(disposition)
    const filename = matched?.[1] ?? 'agent-turns.csv'
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    exporting.value = false
  }
}

function changePage(next: number): void {
  const totalPages = turns.value
    ? Math.max(1, Math.ceil(turns.value.total / pageSize.value))
    : 1
  page.value = Math.min(Math.max(1, next), totalPages)
  void loadAll()
}

const totalPages = computed(() =>
  turns.value ? Math.max(1, Math.ceil(turns.value.total / pageSize.value)) : 1,
)

function reasonLabel(code: string): string {
  return REASON_LABELS[code] ?? code
}

async function showFeedbackTurns(rating: 'like' | 'dislike'): Promise<void> {
  filters.rating = rating
  page.value = 1
  await loadAll()
  await nextTick()
  turnsSection.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const overviewCards = computed(() => {
  const item = metrics.value?.overview
  if (!item) {
    return []
  }
  return [
    { label: '会话数', value: String(item.sessions) },
    { label: '总轮次', value: String(item.turns) },
    { label: '任务型轮次', value: String(item.task_turns) },
    { label: '工具调用', value: String(item.tool_calls) },
    { label: '工具成功率', value: fmtPercent(item.tool_success_rate) },
    { label: '平均轮次耗时', value: fmtMs(item.avg_turn_latency_ms) },
    { label: 'P95 轮次耗时', value: fmtMs(item.p95_turn_latency_ms) },
    { label: '总 token', value: item.total_tokens.toLocaleString('zh-CN') },
    { label: '赞 / 踩', value: `${item.likes} / ${item.dislikes}` },
    { label: '满意率', value: fmtPercent(item.satisfaction_rate) },
    { label: '告警数', value: String(item.alerts) },
  ]
})

function feedbackLabel(rate: number | null): string {
  return fmtPercent(rate)
}

onMounted(async () => {
  const config = await loadAppConfig()
  pageSize.value = Math.max(1, config.admin_page_size)
  await loadAll()
})
</script>

<template>
  <div class="admin-shell">
    <header class="admin-header">
      <div class="admin-brand">
        <div class="brand-mark">A</div>
        <div>
          <h1>管理后台</h1>
          <p>工具、模型、调用链与满意度</p>
        </div>
      </div>
      <div class="admin-header-actions">
        <span class="admin-warning">仅限本地使用，公网部署前必须加鉴权</span>
        <router-link class="ghost-btn" to="/">返回聊天</router-link>
      </div>
    </header>

    <div class="admin-toolbar">
      <label>
        开始
        <input v-model="start" type="datetime-local" />
      </label>
      <label>
        结束
        <input v-model="end" type="datetime-local" />
      </label>
      <button class="primary-btn" type="button" :disabled="loading" @click="applyFilters">
        {{ loading ? '加载中…' : '刷新' }}
      </button>
      <button class="ghost-btn" type="button" @click="quickRange(24)">近 24 小时</button>
      <button class="ghost-btn" type="button" @click="quickRange(24 * 7)">近 7 天</button>
      <button class="ghost-btn" type="button" @click="quickRange(24 * 30)">近 30 天</button>
      <button class="ghost-btn" type="button" :disabled="exporting" @click="exportCsv">
        {{ exporting ? '导出中…' : '导出 CSV' }}
      </button>
      <span class="admin-range-note">时区 Asia/Shanghai · 最大 90 天 · 分页 {{ pageSize }}</span>
    </div>

    <div v-if="error" class="admin-error">{{ error }}</div>

    <div v-if="metrics?.alerts.length" class="alert-bar">
      <strong>异常告警</strong>
      <button
        v-for="alert in metrics.alerts"
        :key="alert.rule"
        class="alert-item"
        type="button"
        @click="applyAlert(alert.filters)"
      >
        {{ alert.name }}：{{ alert.current }}（阈值 {{ alert.threshold }}）· {{ alert.window }}
      </button>
    </div>

    <div v-if="loading && !metrics" class="admin-empty">正在加载后台数据…</div>
    <div v-else-if="!metrics" class="admin-empty">所选范围内暂无数据</div>

    <template v-if="metrics">
      <section class="overview-grid">
        <article v-for="card in overviewCards" :key="card.label" class="overview-card">
          <span>{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
        </article>
      </section>

      <section class="admin-section">
        <header class="section-head">
          <h2>趋势</h2>
          <span>{{ metrics.range.granularity === 'hour' ? '按小时' : '按天' }}</span>
        </header>
        <AdminTrend :points="metrics.trend" />
      </section>

      <nav class="admin-tabs">
        <button :class="{ active: tab === 'tools' }" type="button" @click="tab = 'tools'">工具</button>
        <button :class="{ active: tab === 'models' }" type="button" @click="tab = 'models'">模型</button>
        <button :class="{ active: tab === 'context' }" type="button" @click="tab = 'context'">
          上下文
        </button>
        <button :class="{ active: tab === 'sessions' }" type="button" @click="tab = 'sessions'">
          会话
        </button>
        <button
          :class="{ active: tab === 'satisfaction' }"
          type="button"
          @click="tab = 'satisfaction'"
        >
          评价
        </button>
      </nav>

      <section v-if="tab === 'tools'" class="admin-section">
        <div v-if="!metrics.tools.length" class="admin-empty">暂无工具调用</div>
        <table v-else class="admin-table">
          <thead>
            <tr>
              <th>工具</th>
              <th>调用</th>
              <th>成功</th>
              <th>失败</th>
              <th>禁用</th>
              <th>成功率</th>
              <th>平均耗时</th>
              <th>P95</th>
              <th>重试</th>
              <th>Top 失败原因</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="tool in metrics.tools"
              :key="tool.tool_name"
              class="clickable-row"
              @click="filterByTool(tool.tool_name)"
            >
              <td>{{ tool.tool_name }}</td>
              <td>{{ tool.calls }}</td>
              <td>{{ tool.ok }}</td>
              <td>{{ tool.error }}</td>
              <td>{{ tool.blocked }}</td>
              <td>{{ fmtPercent(tool.success_rate) }}</td>
              <td>{{ fmtMs(tool.avg_ms) }}</td>
              <td>{{ fmtMs(tool.p95_ms) }}</td>
              <td>{{ tool.retries }}</td>
              <td>
                <span v-if="!tool.top_errors.length">—</span>
                <span v-for="item in tool.top_errors" :key="item.reason" class="top-error">
                  {{ item.reason }} × {{ item.count }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <section v-else-if="tab === 'models'" class="admin-section">
        <div v-if="!metrics.models.length" class="admin-empty">暂无模型调用</div>
        <table v-else class="admin-table">
          <thead>
            <tr>
              <th>模型层</th>
              <th>调用</th>
              <th>成功</th>
              <th>失败</th>
              <th>降级</th>
              <th>熔断</th>
              <th>半开</th>
              <th>探测成功</th>
              <th>探测失败</th>
              <th>平均耗时</th>
              <th>token</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="model in metrics.models" :key="model.layer">
              <td>{{ model.layer }}</td>
              <td>{{ model.calls }}</td>
              <td>{{ model.ok }}</td>
              <td>{{ model.error }}</td>
              <td>{{ model.degrade }}</td>
              <td>{{ model.breaker_open }}</td>
              <td>{{ model.half_open }}</td>
              <td>{{ model.probe_success }}</td>
              <td>{{ model.probe_failure }}</td>
              <td>{{ fmtMs(model.avg_ms) }}</td>
              <td>{{ model.tokens.toLocaleString('zh-CN') }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section v-else-if="tab === 'context'" class="admin-section">
        <div class="metric-grid">
          <article><span>压缩次数</span><strong>{{ metrics.context.compress_count }}</strong></article>
          <article>
            <span>压缩前平均 token</span>
            <strong>{{ metrics.context.tokens_before_avg ?? '—' }}</strong>
          </article>
          <article>
            <span>压缩后平均 token</span>
            <strong>{{ metrics.context.tokens_after_avg ?? '—' }}</strong>
          </article>
          <article>
            <span>压缩前总量</span>
            <strong>{{ metrics.context.tokens_before_total.toLocaleString('zh-CN') }}</strong>
          </article>
          <article>
            <span>压缩后总量</span>
            <strong>{{ metrics.context.tokens_after_total.toLocaleString('zh-CN') }}</strong>
          </article>
          <article>
            <span>预算收束次数</span>
            <strong>{{ metrics.context.budget_wrap_count }}</strong>
          </article>
          <article>
            <span>30 轮触顶</span>
            <strong>{{ metrics.context.max_turn_wrap_count }}</strong>
          </article>
          <article>
            <span>native 切换</span>
            <strong>{{ metrics.context.mode_switch_count }}</strong>
          </article>
        </div>
      </section>

      <section v-else-if="tab === 'sessions'" class="admin-section">
        <div class="metric-grid">
          <article><span>会话数</span><strong>{{ metrics.sessions.count }}</strong></article>
          <article><span>平均任务轮次</span><strong>{{ metrics.sessions.avg_task_turns }}</strong></article>
          <article><span>30 轮触顶会话</span><strong>{{ metrics.sessions.wrapped }}</strong></article>
          <article><span>native 会话</span><strong>{{ metrics.sessions.native }}</strong></article>
        </div>
        <table class="admin-table compact">
          <thead>
            <tr>
              <th>任务轮次分布</th>
              <th v-for="(_value, key) in metrics.sessions.distribution" :key="key">{{ key }}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>会话数</td>
              <td v-for="(value, key) in metrics.sessions.distribution" :key="key">{{ value }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section v-else class="admin-section">
        <div class="metric-grid">
          <article><span>评价总数</span><strong>{{ metrics.satisfaction.total }}</strong></article>
          <article><span>赞</span><strong>{{ metrics.satisfaction.likes }}</strong></article>
          <article><span>踩</span><strong>{{ metrics.satisfaction.dislikes }}</strong></article>
          <article><span>满意率</span><strong>{{ fmtPercent(metrics.satisfaction.rate) }}</strong></article>
        </div>
        <div class="satisfaction-links">
          <button class="ghost-btn" type="button" @click="showFeedbackTurns('like')">
            查看赞评价明细
          </button>
          <button class="ghost-btn" type="button" @click="showFeedbackTurns('dislike')">
            查看踩评价明细
          </button>
        </div>
        <div class="split-tables">
          <div>
            <h3>原因分布</h3>
            <div v-if="!metrics.satisfaction.reasons.length" class="admin-empty">暂无评价原因</div>
            <table v-else class="admin-table compact">
              <thead><tr><th>原因</th><th>数量</th></tr></thead>
              <tbody>
                <tr v-for="item in metrics.satisfaction.reasons" :key="item.code">
                  <td>{{ reasonLabel(item.code) }}</td>
                  <td>{{ item.count }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <h3>按工具交叉</h3>
            <div v-if="!metrics.satisfaction.by_tool.length" class="admin-empty">暂无数据</div>
            <table v-else class="admin-table compact">
              <thead><tr><th>工具</th><th>赞</th><th>踩</th><th>满意率</th></tr></thead>
              <tbody>
                <tr v-for="item in metrics.satisfaction.by_tool" :key="item.tool_name">
                  <td>{{ item.tool_name }}</td>
                  <td>{{ item.likes }}</td>
                  <td>{{ item.dislikes }}</td>
                  <td>{{ feedbackLabel(item.rate) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <h3>按模型层交叉</h3>
            <div v-if="!metrics.satisfaction.by_model.length" class="admin-empty">暂无数据</div>
            <table v-else class="admin-table compact">
              <thead><tr><th>模型层</th><th>赞</th><th>踩</th><th>满意率</th></tr></thead>
              <tbody>
                <tr v-for="item in metrics.satisfaction.by_model" :key="item.layer">
                  <td>{{ item.layer }}</td>
                  <td>{{ item.likes }}</td>
                  <td>{{ item.dislikes }}</td>
                  <td>{{ feedbackLabel(item.rate) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section ref="turnsSection" class="admin-section">
        <header class="section-head">
          <h2>调用链明细</h2>
          <span>共 {{ turns?.total ?? 0 }} 条</span>
        </header>
        <div class="admin-filters">
          <input v-model="filters.session_id" placeholder="session_id" />
          <select v-model="filters.tool_name">
            <option value="">全部工具</option>
            <option v-for="name in toolOptions" :key="name" :value="name">{{ name }}</option>
          </select>
          <select v-model="filters.tool_status">
            <option value="">全部状态</option>
            <option value="ok">成功</option>
            <option value="error">失败</option>
            <option value="blocked">禁用</option>
          </select>
          <select v-model="filters.rating">
            <option value="">全部评价</option>
            <option value="any">有评价</option>
            <option value="none">无评价</option>
            <option value="like">赞</option>
            <option value="dislike">踩</option>
          </select>
          <select v-model="filters.finish_reason">
            <option value="">全部结束原因</option>
            <option v-for="item in finishOptions" :key="item.value" :value="item.value">
              {{ item.label }}
            </option>
          </select>
          <input v-model="filters.trace_id" placeholder="trace_id" />
          <button class="primary-btn" type="button" @click="applyFilters">应用筛选</button>
          <button class="ghost-btn" type="button" @click="resetFilters">重置</button>
        </div>

        <div v-if="!turns?.items.length" class="admin-empty">暂无调用链数据</div>
        <table v-else class="admin-table turns-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>trace_id</th>
              <th>会话 / 轮次</th>
              <th>用户输入</th>
              <th>最终回答</th>
              <th>模型层</th>
              <th>工具链</th>
              <th>耗时</th>
              <th>token</th>
              <th>评价</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="turn in turns.items"
              :key="turn.trace_id"
              class="clickable-row"
              @click="openTurn(turn.trace_id)"
            >
              <td>{{ fmtDateTime(turn.started_at) }}</td>
              <td><code class="trace-cell" :title="turn.trace_id">{{ turn.trace_id.slice(0, 10) }}…</code></td>
              <td>
                <div class="cell-strong">{{ turn.session_title || '未命名会话' }}</div>
                <div class="cell-muted" :title="turn.session_id">{{ turn.session_id }}</div>
                <div class="cell-muted">#{{ turn.turn_no }}</div>
              </td>
              <td class="cell-text">{{ turn.user_input }}</td>
              <td class="cell-text">{{ turn.final_answer }}</td>
              <td>{{ turn.model_layer || '—' }}</td>
              <td>
                <span v-for="tool in turn.tools" :key="tool.id" class="tool-chip" :class="tool.status">
                  {{ tool.name }} · {{ tool.status }} · {{ tool.latency_ms }}ms
                </span>
                <span v-if="!turn.tools.length" class="cell-muted">—</span>
              </td>
              <td>{{ fmtMs(turn.duration_ms) }}</td>
              <td>{{ turn.tokens }}</td>
              <td>
                <span v-if="turn.feedback?.rating === 'like'" class="rate-like">赞</span>
                <span v-else-if="turn.feedback?.rating === 'dislike'" class="rate-dislike">踩</span>
                <span v-else class="cell-muted">—</span>
              </td>
            </tr>
          </tbody>
        </table>

        <div v-if="turns" class="pager">
          <button class="ghost-btn" type="button" :disabled="page <= 1" @click="changePage(page - 1)">
            上一页
          </button>
          <span>第 {{ page }} / {{ totalPages }} 页</span>
          <button
            class="ghost-btn"
            type="button"
            :disabled="page >= totalPages"
            @click="changePage(page + 1)"
          >
            下一页
          </button>
        </div>
      </section>
    </template>
  </div>
</template>
