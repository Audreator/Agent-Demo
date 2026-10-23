<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, isHttpUrl, loadAppConfig } from '@/api/client'
import type { Citation } from '@/types'

const props = defineProps<{
  citations: Citation[]
  sessionId: string
  turnNo?: number
}>()

const expanded = ref(false)
const failedOpen = ref(false)
const checkingId = ref('')
const overrides = ref<Record<string, Citation>>({})
const error = ref('')
const displayLimit = ref(3)

onMounted(async () => {
  const config = await loadAppConfig()
  displayLimit.value = Math.max(1, config.citation_display_limit)
})

watch(
  () => props.citations,
  () => {
    overrides.value = {}
  },
)

function statusOf(item: Citation): Citation {
  return overrides.value[item.id] ?? item
}

const available = computed(() =>
  props.citations.map(statusOf).filter((item) => item.reachable && isHttpUrl(item.url)),
)
const failed = computed(() =>
  props.citations.map(statusOf).filter((item) => !item.reachable || !isHttpUrl(item.url)),
)
const shown = computed(() =>
  expanded.value ? available.value : available.value.slice(0, displayLimit.value),
)

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return ''
  }
}

function failedLabel(item: Citation): string {
  if (!isHttpUrl(item.url)) {
    return '链接格式不允许'
  }
  if (item.status_code) {
    const reason =
      item.reason && !item.reason.startsWith('HTTP') ? ` · ${item.reason}` : ''
    return `HTTP ${item.status_code}${reason}`
  }
  return item.reason || '未通过校验'
}

function navigate(url: string): void {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.target = '_blank'
  anchor.rel = 'noopener noreferrer'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

async function openCitation(item: Citation): Promise<void> {
  const current = statusOf(item)
  if (!current.reachable || !isHttpUrl(current.url) || checkingId.value) {
    return
  }
  checkingId.value = item.id
  error.value = ''
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 2000)
  try {
    const response = await api<{ citations: Citation[] }>('/api/citations/check', {
      method: 'POST',
      body: JSON.stringify({
        session_id: props.sessionId,
        turn_no: props.turnNo,
        citation_ids: [item.id],
      }),
      signal: controller.signal,
    })
    const updated = response.citations[0]
    if (updated) {
      overrides.value = { ...overrides.value, [item.id]: updated }
      if (!updated.reachable) {
        error.value = `链接未通过校验${updated.reason ? `：${updated.reason}` : ''}`
        return
      }
    }
    navigate(current.url)
  } catch (err) {
    const aborted =
      (err instanceof DOMException && err.name === 'AbortError') ||
      (err instanceof Error && err.name === 'AbortError')
    const message = err instanceof Error ? err.message : String(err)
    const statusMatch = /\((\d{3})\)/.exec(message)
    const status = statusMatch ? Number(statusMatch[1]) : 0
    if (aborted || !status || status >= 500) {
      navigate(current.url)
    } else {
      error.value = message
    }
  } finally {
    window.clearTimeout(timer)
    checkingId.value = ''
  }
}
</script>

<template>
  <section class="citation-list">
    <header class="citation-head">
      <span class="citation-icon">↗</span>
      <strong>搜索来源</strong>
      <span class="citation-count">{{ available.length }} 条可访问</span>
    </header>

    <div v-if="!available.length" class="citation-empty">未获取到可访问来源</div>

    <a
      v-for="item in shown"
      :key="item.id"
      class="citation-item"
      :href="item.url"
      target="_blank"
      rel="noopener noreferrer"
      @click.prevent="openCitation(item)"
    >
      <span class="citation-title">{{ item.title || item.url }}</span>
      <span class="citation-domain">{{ hostOf(item.url) }}</span>
      <p v-if="item.snippet" class="citation-snippet">{{ item.snippet }}</p>
      <span class="citation-badge" :class="{ checking: checkingId === item.id }">
        {{ checkingId === item.id ? '校验中…' : '可访问' }}
      </span>
    </a>

    <button
      v-if="available.length > displayLimit"
      class="citation-toggle"
      type="button"
      @click="expanded = !expanded"
    >
      {{ expanded ? '收起' : `展开全部 ${available.length} 条` }}
    </button>

    <div v-if="error" class="citation-error">{{ error }}</div>

    <div v-if="failed.length" class="citation-failed">
      <button class="citation-toggle muted" type="button" @click="failedOpen = !failedOpen">
        {{ failedOpen ? '收起' : `${failed.length} 条链接未通过校验` }}
      </button>
      <ul v-show="failedOpen" class="citation-failed-list">
        <li v-for="item in failed" :key="`failed-${item.id}`" class="citation-failed-item">
          <span class="citation-title">{{ item.title || item.url }}</span>
          <span class="citation-domain">{{ hostOf(item.url) }}</span>
          <span class="citation-failed-reason">{{ failedLabel(item) }}</span>
        </li>
      </ul>
    </div>
  </section>
</template>
