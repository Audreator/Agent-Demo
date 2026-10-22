<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import ThoughtPanel from './ThoughtPanel.vue'
import ToolCallCard from './ToolCallCard.vue'
import type { ChatMessage, ToolCallView } from '@/types'

const props = defineProps<{
  messages: ChatMessage[]
  sending: boolean
}>()

const scroller = ref<HTMLElement | null>(null)
const pinned = ref(true)

const displayMessages = computed(() =>
  props.messages.filter((message) => {
    if (message.role === 'assistant' && !message.content && !message.thought?.trim()) {
      return Boolean(message.errorNote || message.tools?.length || message.streaming || message.contextNote)
    }
    return true
  }),
)

function toolFromMessage(message: ChatMessage): ToolCallView {
  const meta = message.meta
  return {
    id: String(meta.tool_call_id ?? message.id),
    name: String(meta.tool_name ?? 'tool'),
    arguments: (meta.arguments as Record<string, unknown> | undefined) ?? {},
    status: meta.status === 'ok' ? 'ok' : 'error',
    summary: String(meta.summary ?? message.content),
    latency_ms: typeof meta.latency_ms === 'number' ? meta.latency_ms : undefined,
    result_ref: (meta.result_ref as string | null | undefined) ?? null,
  }
}

function onScroll(): void {
  const element = scroller.value
  if (!element) {
    return
  }
  pinned.value = element.scrollHeight - element.scrollTop - element.clientHeight < 80
}

watch(
  () => props.messages.map((item) => `${item.id}:${item.content.length}:${item.thought?.length ?? 0}:${item.tools?.length ?? 0}`).join('|'),
  async () => {
    if (!pinned.value) {
      return
    }
    await nextTick()
    const element = scroller.value
    if (element) {
      element.scrollTop = element.scrollHeight
    }
  },
)
</script>

<template>
  <div ref="scroller" class="message-scroll" @scroll="onScroll">
    <div v-if="!displayMessages.length" class="empty-chat">
      <div class="empty-orb">✦</div>
      <h2>开始一段任务对话</h2>
      <p>
        试试：「帮我算一下 0.1 + 0.2，然后建一个待办提醒我晚上复盘」<br />
        Agent 会思考、调用 MCP 工具，并把过程实时展示出来。
      </p>
    </div>

    <div
      v-for="message in displayMessages"
      :key="message.id"
      class="message-row"
      :class="message.role"
    >
      <div v-if="message.role === 'user'" class="bubble user-bubble">
        {{ message.content }}
      </div>

      <div v-else-if="message.role === 'tool'" class="tool-message">
        <ToolCallCard :tool="toolFromMessage(message)" />
      </div>

      <div v-else class="assistant-column">
        <ThoughtPanel
          v-if="message.thought?.trim()"
          :thought="message.thought"
          :streaming="message.streaming"
        />
        <ToolCallCard v-for="tool in message.tools ?? []" :key="tool.id" :tool="tool" />
        <div v-if="message.content" class="bubble assistant-bubble">
          {{ message.content }}<span v-if="message.streaming" class="caret"></span>
        </div>
        <div v-if="message.errorNote" class="inline-alert">
          输出协议异常：{{ message.errorNote }}
        </div>
        <div v-if="message.contextNote" class="context-note">{{ message.contextNote }}</div>
        <div
          v-if="message.streaming && !message.content && !message.thought && !message.tools?.length"
          class="bubble assistant-bubble placeholder"
        >
          <span class="thinking-dots"><i></i><i></i><i></i></span>
        </div>
      </div>
    </div>

    <div v-if="props.sending" class="stream-status">
      <span class="pulse"></span> Agent 正在处理…
    </div>
  </div>
</template>
