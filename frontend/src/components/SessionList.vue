<script setup lang="ts">
import type { SessionInfo } from '@/types'

const props = defineProps<{
  sessions: SessionInfo[]
  currentId: string | null
  disabled: boolean
}>()

const emit = defineEmits<{
  (e: 'select', id: string): void
  (e: 'create'): void
  (e: 'delete', id: string): void
}>()

function formatTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return ''
  }
  const now = Date.now()
  const diff = Math.max(0, now - date.getTime())
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) {
    return '刚刚'
  }
  if (minutes < 60) {
    return `${minutes} 分钟前`
  }
  const hours = Math.floor(minutes / 60)
  if (hours < 24) {
    return `${hours} 小时前`
  }
  return `${Math.floor(hours / 24)} 天前`
}

function confirmDelete(event: MouseEvent, session: SessionInfo): void {
  event.stopPropagation()
  if (window.confirm(`删除会话「${session.title}」？该操作不可恢复。`)) {
    emit('delete', session.id)
  }
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">A</div>
      <div class="brand-text">
        <strong>Minimal Agent</strong>
        <span>ReAct · MCP · SSE</span>
      </div>
    </div>

    <button class="new-session" :disabled="disabled" @click="emit('create')">
      <span class="plus">+</span> 新建会话
    </button>

    <div class="list-label">
      <span>会话</span>
      <span class="count">{{ props.sessions.length }}</span>
    </div>

    <div class="session-list">
      <button
        v-for="session in props.sessions"
        :key="session.id"
        class="session-item"
        :class="{ active: session.id === props.currentId }"
        :disabled="disabled"
        @click="emit('select', session.id)"
      >
        <div class="session-main">
          <span class="session-title">{{ session.title }}</span>
          <span class="session-meta">
            <span class="mode-dot" :class="session.mode">{{ session.mode }}</span>
            {{ formatTime(session.updated_at) }}
          </span>
        </div>
        <span
          v-if="session.id === props.currentId && !disabled"
          class="delete"
          title="删除会话"
          @click="confirmDelete($event, session)"
        >
          ×
        </span>
      </button>
      <div v-if="!props.sessions.length" class="empty-list">
        还没有会话，点击上方按钮开始。
      </div>
    </div>

    <slot name="tools"></slot>

    <div class="sidebar-footer">
      <span class="status-dot" :class="{ on: !disabled }"></span>
      {{ disabled ? 'Agent 正在运行' : 'dev-user · 就绪' }}
    </div>

    <router-link class="admin-entry" to="/admin">管理后台 →</router-link>
  </aside>
</template>
