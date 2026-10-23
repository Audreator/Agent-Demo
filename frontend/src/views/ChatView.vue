<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import SessionList from '@/components/SessionList.vue'
import MessageStream from '@/components/MessageStream.vue'
import ToolPanel from '@/components/ToolPanel.vue'
import { useChatStore } from '@/stores/chat'

const store = useChatStore()
const {
  sessions,
  currentSessionId,
  messages,
  sending,
  loadingMessages,
  connection,
  error,
  activeTraceId,
} = storeToRefs(store)

const input = ref('')
const composing = ref(false)
const textarea = ref<HTMLTextAreaElement | null>(null)

const canSend = computed(
  () =>
    Boolean(currentSessionId.value) &&
    input.value.trim().length > 0 &&
    !sending.value &&
    !loadingMessages.value,
)

const connectionLabel = computed(() => {
  if (connection.value === 'reconnecting') {
    return '重连中'
  }
  if (connection.value === 'streaming') {
    return '流式输出'
  }
  if (connection.value === 'error') {
    return '连接异常'
  }
  return '在线'
})

onMounted(async () => {
  try {
    await store.loadSessions()
  } catch (err) {
    store.error = err instanceof Error ? err.message : String(err)
  }
})

function resize(): void {
  const element = textarea.value
  if (!element) {
    return
  }
  element.style.height = 'auto'
  element.style.height = `${Math.min(element.scrollHeight, 180)}px`
}

async function submit(): Promise<void> {
  const text = input.value.trim()
  if (!canSend.value || composing.value) {
    return
  }
  input.value = ''
  await nextTick()
  resize()
  await store.send(text)
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey && !composing.value) {
    event.preventDefault()
    void submit()
  }
}

async function createSession(): Promise<void> {
  try {
    await store.createSession()
    await nextTick()
    textarea.value?.focus()
  } catch (err) {
    store.error = err instanceof Error ? err.message : String(err)
  }
}
</script>

<template>
  <div class="app-shell">
    <SessionList
      :sessions="sessions"
      :current-id="currentSessionId"
      :disabled="sending"
      @select="store.selectSession"
      @create="createSession"
      @delete="store.deleteSession"
    >
      <template #tools>
        <ToolPanel />
      </template>
    </SessionList>

    <main class="chat-panel">
      <header class="chat-header">
        <div class="chat-title">
          <h1>{{ store.currentSession?.title ?? 'Minimal Agent' }}</h1>
          <div class="chat-sub">
            <span class="badge" :class="store.currentSession?.mode ?? 'react'">
              {{ store.currentSession?.mode === 'native' ? 'Native Tool Calls' : 'Text ReAct' }}
            </span>
            <span v-if="store.currentSession" class="task-count">
              任务轮次 {{ store.currentSession.task_turn_count }} / 30
            </span>
            <span v-if="activeTraceId" class="trace" :title="activeTraceId">
              trace {{ activeTraceId.slice(0, 8) }}
            </span>
          </div>
        </div>
        <div class="conn" :class="connection">
          <span class="conn-dot"></span>{{ connectionLabel }}
        </div>
      </header>

      <div v-if="error" class="error-banner">
        <span>{{ error }}</span>
        <button type="button" @click="store.error = ''">关闭</button>
      </div>

      <MessageStream :messages="messages" :sending="sending" :session-id="currentSessionId" />

      <footer class="composer">
        <div v-if="!currentSessionId" class="composer-hint">
          请先新建或选择一个会话。
        </div>
        <div class="composer-box" :class="{ disabled: !currentSessionId }">
          <textarea
            ref="textarea"
            v-model="input"
            rows="1"
            :disabled="!currentSessionId"
            placeholder="输入消息，Enter 发送，Shift + Enter 换行"
            @input="resize"
            @keydown="onKeydown"
            @compositionstart="composing = true"
            @compositionend="composing = false"
          ></textarea>
          <button v-if="sending" class="send stop" type="button" @click="store.stop">停止</button>
          <button v-else class="send" type="button" :disabled="!canSend" @click="submit">
            发送
          </button>
        </div>
        <div class="composer-tip">
          上下文自动压缩 · 三层模型熔断 · 工具调用可折叠 · 断线自动续传
        </div>
      </footer>
    </main>
  </div>
</template>
