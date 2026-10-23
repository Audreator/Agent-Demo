<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api, fmtDateTime, fmtMs } from '@/api/client'
import type { AdminTurnDetail, TraceEventItem } from '@/types'

const route = useRoute()
const traceId = computed(() => String(route.params.traceId ?? ''))
const detail = ref<AdminTurnDetail | null>(null)
const loading = ref(false)
const error = ref('')
const expandedPayloads = reactive<Record<string, boolean>>({})
const copied = ref(false)

const depthMap = computed(() => {
  const events = detail.value?.events ?? []
  const bySpan = new Map(events.map((event) => [event.span_id, event]))
  const result: Record<string, number> = {}
  for (const event of events) {
    let depth = 0
    let parent = event.parent_span_id
    const seen = new Set<string>()
    while (parent && depth < 12 && !seen.has(parent)) {
      seen.add(parent)
      depth += 1
      parent = bySpan.get(parent)?.parent_span_id ?? null
    }
    result[event.span_id] = depth
  }
  return result
})

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    detail.value = await api<AdminTurnDetail>(`/api/admin/turns/${traceId.value}`)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function payloadText(event: TraceEventItem): string {
  try {
    const text = JSON.stringify(event.payload, null, 2)
    return text.length > 2000 ? `${text.slice(0, 2000)}…` : text
  } catch {
    return String(event.payload)
  }
}

function payloadSummary(event: TraceEventItem): string {
  const payload = event.payload ?? {}
  const parts: string[] = []
  for (const key of ['provider', 'layer', 'tool_name', 'status', 'error', 'reason', 'tokens_before', 'tokens_after']) {
    if (payload[key] !== undefined && payload[key] !== null) {
      parts.push(`${key}=${String(payload[key]).slice(0, 60)}`)
    }
  }
  if (!parts.length) {
    const text = payloadText(event)
    parts.push(text.replace(/\s+/g, ' ').slice(0, 100))
  }
  return parts.join(' · ')
}

async function copyTrace(): Promise<void> {
  try {
    await navigator.clipboard.writeText(traceId.value)
    copied.value = true
    window.setTimeout(() => {
      copied.value = false
    }, 1500)
  } catch {
    // 忽略剪贴板不可用
  }
}

onMounted(() => {
  void load()
})
</script>

<template>
  <div class="admin-shell">
    <header class="admin-header">
      <div class="admin-brand">
        <div class="brand-mark">A</div>
        <div>
          <h1>调用链详情</h1>
          <p>trace {{ traceId }}</p>
        </div>
      </div>
      <div class="admin-header-actions">
        <span class="admin-warning">仅限本地使用，公网部署前必须加鉴权</span>
        <button class="ghost-btn" type="button" @click="copyTrace">
          {{ copied ? '已复制' : '复制 trace_id' }}
        </button>
        <a
          class="ghost-btn"
          :href="`/api/traces/${traceId}`"
          target="_blank"
          rel="noopener noreferrer"
        >
          原始 JSON
        </a>
        <router-link class="ghost-btn" to="/admin">返回后台</router-link>
        <router-link class="ghost-btn" to="/">返回聊天</router-link>
      </div>
    </header>

    <div v-if="loading" class="admin-empty">正在加载调用链…</div>
    <div v-else-if="error" class="admin-error">{{ error }}</div>
    <div v-else-if="!detail" class="admin-empty">调用链不存在</div>

    <template v-else>
      <section class="overview-grid">
        <article class="overview-card">
          <span>轮次</span><strong>#{{ detail.summary.turn_no }}</strong>
        </article>
        <article class="overview-card">
          <span>模型层</span><strong>{{ detail.summary.model_layer || '—' }}</strong>
        </article>
        <article class="overview-card">
          <span>轮次耗时</span><strong>{{ fmtMs(detail.summary.duration_ms) }}</strong>
        </article>
        <article class="overview-card">
          <span>token</span><strong>{{ detail.summary.tokens }}</strong>
        </article>
        <article class="overview-card">
          <span>结束原因</span><strong>{{ detail.summary.finish_reason }}</strong>
        </article>
        <article class="overview-card">
          <span>评价</span>
          <strong>
            {{ detail.summary.feedback?.rating === 'like' ? '赞' : detail.summary.feedback?.rating === 'dislike' ? '踩' : '—' }}
          </strong>
        </article>
      </section>

      <section class="admin-section">
        <header class="section-head"><h2>消息</h2></header>
        <div class="detail-messages">
          <article v-for="message in detail.messages" :key="message.id" class="detail-message">
            <div class="detail-message-head">
              <span class="role-chip" :class="message.role">{{ message.role }}</span>
              <span>{{ fmtDateTime(message.created_at) }}</span>
            </div>
            <pre>{{ message.content }}</pre>
          </article>
        </div>
      </section>

      <section class="admin-section">
        <header class="section-head"><h2>工具调用</h2></header>
        <div v-if="!detail.tool_calls.length" class="admin-empty">本轮无工具调用</div>
        <table v-else class="admin-table">
          <thead>
            <tr>
              <th>工具</th>
              <th>状态</th>
              <th>重试</th>
              <th>耗时</th>
              <th>参数</th>
              <th>摘要</th>
              <th>错误</th>
              <th>引用</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="tool in detail.tool_calls" :key="tool.id">
              <td>{{ tool.name }}</td>
              <td><span class="tool-chip" :class="tool.status">{{ tool.status }}</span></td>
              <td>{{ tool.retry_count }}</td>
              <td>{{ fmtMs(tool.latency_ms) }}</td>
              <td><pre class="cell-pre">{{ JSON.stringify(tool.arguments, null, 2) }}</pre></td>
              <td class="cell-text">{{ tool.summary || '—' }}</td>
              <td class="cell-text">{{ tool.error || '—' }}</td>
              <td class="cell-text">{{ tool.result_ref || '—' }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="admin-section">
        <header class="section-head">
          <h2>Trace 时间线</h2>
          <span>{{ detail.events.length }} 个 span</span>
        </header>
        <div class="trace-tree">
          <article
            v-for="event in detail.events"
            :key="event.id"
            class="trace-node"
            :style="{ marginLeft: `${(depthMap[event.span_id] ?? 0) * 18}px` }"
          >
            <div class="trace-node-head">
              <span class="trace-kind">{{ event.kind }}</span>
              <span class="trace-duration">{{ fmtMs(event.duration_ms) }}</span>
              <span class="trace-span">{{ event.span_id }}</span>
              <span class="trace-parent">← {{ event.parent_span_id || 'root' }}</span>
              <span class="cell-muted">{{ fmtDateTime(event.ts) }}</span>
            </div>
            <p class="trace-summary">{{ payloadSummary(event) }}</p>
            <button
              class="ghost-btn small"
              type="button"
              @click="expandedPayloads[String(event.id)] = !expandedPayloads[String(event.id)]"
            >
              {{ expandedPayloads[String(event.id)] ? '收起 payload' : '展开 payload' }}
            </button>
            <pre v-if="expandedPayloads[String(event.id)]" class="trace-payload">{{
              payloadText(event)
            }}</pre>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>
