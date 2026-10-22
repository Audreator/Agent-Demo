<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ToolCallView } from '@/types'

const props = defineProps<{
  tool: ToolCallView
}>()

const open = ref(false)

const statusText = computed(() => {
  switch (props.tool.status) {
    case 'running':
      return '执行中'
    case 'ok':
      return '成功'
    case 'blocked':
      return '已禁用'
    default:
      return '失败'
  }
})

const prettyArgs = computed(() => {
  try {
    return JSON.stringify(props.tool.arguments, null, 2)
  } catch {
    return String(props.tool.arguments)
  }
})
</script>

<template>
  <div class="tool-card" :class="props.tool.status">
    <button class="tool-head" @click="open = !open">
      <span class="tool-glyph">⚙</span>
      <span class="tool-name">{{ props.tool.name }}</span>
      <span class="tool-status" :class="props.tool.status">
        <i v-if="props.tool.status === 'running'" class="spinner"></i>
        {{ statusText }}
      </span>
      <span v-if="props.tool.latency_ms !== undefined" class="tool-latency">
        {{ props.tool.latency_ms }}ms
      </span>
      <span class="tool-toggle">{{ open ? '−' : '+' }}</span>
    </button>

    <div v-if="props.tool.summary || props.tool.error" class="tool-summary">
      {{ props.tool.error || props.tool.summary }}
    </div>

    <div v-show="open" class="tool-detail">
      <div class="tool-section">
        <span class="tool-section-label">参数</span>
        <pre>{{ prettyArgs }}</pre>
      </div>
      <div v-if="props.tool.summary" class="tool-section">
        <span class="tool-section-label">结果</span>
        <pre>{{ props.tool.summary }}</pre>
      </div>
      <div v-if="props.tool.result_ref" class="tool-section">
        <span class="tool-section-label">引用</span>
        <code>{{ props.tool.result_ref }}</code>
      </div>
    </div>
  </div>
</template>
