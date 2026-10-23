<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api } from '@/api/client'
import type { ToolInfo } from '@/types'

const tools = ref<ToolInfo[]>([])
const collapsed = ref(false)
const loading = ref(false)
const error = ref('')
let timer: number | undefined

const online = computed(() => tools.value.length > 0 && tools.value.every((item) => item.online))

async function refresh(): Promise<void> {
  loading.value = true
  try {
    tools.value = await api<ToolInfo[]>('/api/tools')
    error.value = ''
  } catch (err) {
    tools.value = tools.value.map((item) => ({ ...item, online: false }))
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  void refresh()
  timer = window.setInterval(() => void refresh(), 30_000)
})

onUnmounted(() => {
  if (timer !== undefined) {
    window.clearInterval(timer)
  }
})
</script>

<template>
  <section class="tool-panel" :class="{ offline: !online }">
    <button class="tool-panel-head" type="button" @click="collapsed = !collapsed">
      <span class="tool-panel-title">
        <span class="online-dot" :class="{ on: online }"></span>
        在线工具
        <span class="tool-count">{{ tools.length }}</span>
      </span>
      <span class="panel-toggle">{{ collapsed ? '+' : '−' }}</span>
    </button>

    <div v-show="!collapsed" class="tool-panel-body">
      <div v-if="!online && (!loading || tools.length)" class="tool-offline-note">工具服务离线</div>
      <div v-if="loading && !tools.length" class="tool-panel-empty">正在检测工具服务…</div>
      <ul v-else class="tool-panel-list">
        <li v-for="tool in tools" :key="tool.name" class="tool-panel-item" :title="tool.description">
          <span class="online-dot small" :class="{ on: tool.online }"></span>
          <div class="tool-panel-main">
            <span class="tool-panel-name">{{ tool.name }}</span>
            <span class="tool-panel-desc">{{ tool.description }}</span>
          </div>
        </li>
      </ul>
      <div v-if="error" class="tool-panel-error">{{ error }}</div>
      <button class="tool-refresh" type="button" :disabled="loading" @click="refresh">
        {{ loading ? '刷新中…' : '刷新工具状态' }}
      </button>
    </div>
  </section>
</template>
