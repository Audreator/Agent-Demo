<script setup lang="ts">
import { computed, ref } from 'vue'
import { api } from '@/api/client'
import type { Citation, ToolCallApi } from '@/types'

const props = defineProps<{
  tools: ToolCallApi[]
  citations?: Citation[]
  sessionId: string
}>()

const open = ref(false)
const modalOpen = ref(false)
const modalLoading = ref(false)
const modalError = ref('')
const modalContent = ref('')
const modalName = ref('')

const successCount = computed(() => props.tools.filter((item) => item.status === 'ok').length)
const errorCount = computed(() => props.tools.filter((item) => item.status === 'error').length)
const blockedCount = computed(() => props.tools.filter((item) => item.status === 'blocked').length)
const totalLatency = computed(() =>
  props.tools.reduce((sum, item) => sum + (item.latency_ms ?? 0), 0),
)
const hasSearch = computed(() => props.tools.some((item) => item.tool_name === 'search'))
const sourceCount = computed(() => props.citations?.length ?? 0)

function statusText(status: string): string {
  switch (status) {
    case 'ok':
      return '成功'
    case 'blocked':
      return '已禁用'
    case 'running':
      return '执行中'
    default:
      return '失败'
  }
}

const SENSITIVE = /token|secret|password|passwd|api[_-]?key|authorization|cookie/i

function maskValue(value: unknown, key: string): unknown {
  if (SENSITIVE.test(key)) {
    return '***'
  }
  if (Array.isArray(value)) {
    return value.map((entry) => maskValue(entry, ''))
  }
  if (value && typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const [childKey, childValue] of Object.entries(value as Record<string, unknown>)) {
      result[childKey] = maskValue(childValue, childKey)
    }
    return result
  }
  return value
}

function maskArgs(argumentsValue: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(argumentsValue ?? {})) {
    result[key] = maskValue(value, key)
  }
  return result
}

function prettyArgs(tool: ToolCallApi): string {
  try {
    const text = JSON.stringify(maskArgs(tool.arguments), null, 2)
    return text.length > 500 ? `${text.slice(0, 500)}…` : text
  } catch {
    return String(tool.arguments).slice(0, 500)
  }
}

async function viewResult(tool: ToolCallApi): Promise<void> {
  modalOpen.value = true
  modalLoading.value = true
  modalError.value = ''
  modalContent.value = ''
  modalName.value = tool.tool_name
  try {
    const response = await api<{ content: string }>(
      `/api/sessions/${props.sessionId}/tool-calls/${tool.id}/result`,
    )
    modalContent.value = response.content
  } catch (err) {
    modalError.value = err instanceof Error ? err.message : String(err)
  } finally {
    modalLoading.value = false
  }
}

async function copyContent(): Promise<void> {
  try {
    await navigator.clipboard.writeText(modalContent.value)
  } catch {
    // 剪贴板不可用时保持原文展示
  }
}

function closeModal(): void {
  modalOpen.value = false
}
</script>

<template>
  <section class="turn-tool-summary">
    <button class="summary-head" type="button" @click="open = !open">
      <span class="summary-icon">⚙</span>
      <span>
        调用了 {{ tools.length }} 个工具 · 成功 {{ successCount }} / 失败 {{ errorCount }} /
        禁用 {{ blockedCount }} · 总耗时 {{ totalLatency }} ms
        <span v-if="hasSearch"> · 来源 {{ sourceCount }} 条</span>
      </span>
      <span class="summary-toggle">{{ open ? '−' : '+' }}</span>
    </button>

    <div v-show="open" class="summary-body">
      <article v-for="tool in tools" :key="tool.id" class="summary-item">
        <header class="summary-item-head">
          <span class="summary-name">{{ tool.tool_name }}</span>
          <span class="tool-status" :class="tool.status">{{ statusText(tool.status) }}</span>
          <span class="summary-meta">
            {{ tool.latency_ms }} ms · 重试 {{ tool.retry_count }} 次
          </span>
        </header>
        <div class="summary-section">
          <span class="summary-label">参数</span>
          <pre>{{ prettyArgs(tool) }}</pre>
        </div>
        <div v-if="tool.result_summary" class="summary-section">
          <span class="summary-label">结果摘要</span>
          <p>{{ tool.result_summary }}</p>
        </div>
        <div v-if="tool.error" class="summary-section error">
          <span class="summary-label">错误</span>
          <p>{{ tool.error }}</p>
        </div>
        <div class="summary-actions">
          <code v-if="tool.result_ref">{{ tool.result_ref }}</code>
          <button
            v-if="tool.status !== 'blocked' && tool.status !== 'running'"
            class="ghost-btn"
            type="button"
            @click="viewResult(tool)"
          >
            查看完整结果
          </button>
        </div>
      </article>
    </div>

    <div v-if="modalOpen" class="modal-mask" @click.self="closeModal">
      <div class="modal-card">
        <header class="modal-head">
          <strong>{{ modalName }} · 完整结果</strong>
          <div class="modal-head-actions">
            <button class="ghost-btn" type="button" :disabled="modalLoading" @click="copyContent">
              复制
            </button>
            <button class="ghost-btn" type="button" @click="closeModal">关闭</button>
          </div>
        </header>
        <div v-if="modalLoading" class="modal-body">加载中…</div>
        <div v-else-if="modalError" class="modal-body modal-error">{{ modalError }}</div>
        <pre v-else class="modal-body">{{ modalContent }}</pre>
      </div>
    </div>
  </section>
</template>
