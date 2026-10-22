<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  thought: string
  streaming?: boolean
}>()

const open = ref(true)
</script>

<template>
  <div class="thought-panel" :class="{ open }">
    <button class="thought-head" @click="open = !open">
      <span class="thought-icon">✦</span>
      <span class="thought-label">思考过程</span>
      <span v-if="props.streaming" class="thinking-dots"><i></i><i></i><i></i></span>
      <span class="chevron">{{ open ? '收起' : '展开' }}</span>
    </button>
    <div v-show="open" class="thought-body">
      <pre>{{ props.thought }}</pre>
    </div>
  </div>
</template>
