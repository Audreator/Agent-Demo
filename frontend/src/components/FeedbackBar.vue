<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { api, loadAppConfig } from '@/api/client'
import type { FeedbackResponse, FeedbackSummary } from '@/types'

const props = defineProps<{
  sessionId: string
  messageId: string
  feedback: FeedbackSummary | null
}>()

const REASONS = [
  { code: 'irrelevant', label: '答非所问' },
  { code: 'tool_error', label: '工具调用错误' },
  { code: 'bad_link', label: '链接失效' },
  { code: 'too_slow', label: '响应太慢' },
  { code: 'fact_error', label: '事实错误' },
  { code: 'incomplete', label: '内容不完整' },
  { code: 'other', label: '其他' },
]

const rating = ref<'like' | 'dislike' | null>(null)
const likes = ref(0)
const dislikes = ref(0)
const panelOpen = ref(false)
const selectedReason = ref('')
const otherText = ref('')
const savedReason = ref<string | null>(null)
const savedReasonText = ref('')
const loading = ref(false)
const error = ref('')
const reasonMax = ref(200)

onMounted(async () => {
  const config = await loadAppConfig()
  reasonMax.value = Math.max(1, config.feedback_reason_max)
})

watch(
  () => props.feedback,
  (value) => {
    rating.value = (value?.rating ?? null) as 'like' | 'dislike' | null
    likes.value = value?.likes ?? 0
    dislikes.value = value?.dislikes ?? 0
    savedReason.value = value?.reason_code ?? null
    savedReasonText.value = value?.reason_text ?? ''
  },
  { immediate: true },
)

function reasonLabel(code: string): string {
  return REASONS.find((item) => item.code === code)?.label ?? code
}

const canSubmitReason = computed(() => {
  if (!selectedReason.value) {
    return false
  }
  if (selectedReason.value === 'other') {
    const text = otherText.value.trim()
    return text.length >= 1 && text.length <= reasonMax.value
  }
  return true
})

function applyResponse(response: FeedbackResponse): void {
  rating.value = response.rating
  likes.value = response.likes
  dislikes.value = response.dislikes
  savedReason.value = response.reason_code
  savedReasonText.value = response.reason_text ?? ''
  panelOpen.value = false
  selectedReason.value = ''
  otherText.value = ''
  error.value = ''
}

async function submitRating(
  next: 'like' | 'dislike',
  reasonCode: string | null = null,
  reasonText: string | null = null,
): Promise<void> {
  if (loading.value) {
    return
  }
  loading.value = true
  error.value = ''
  try {
    const response = await api<FeedbackResponse>(
      `/api/sessions/${props.sessionId}/messages/${props.messageId}/feedback`,
      {
        method: 'POST',
        body: JSON.stringify({
          rating: next,
          reason_code: reasonCode,
          reason_text: reasonText,
        }),
      },
    )
    applyResponse(response)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function cancel(): Promise<void> {
  if (loading.value) {
    return
  }
  loading.value = true
  error.value = ''
  const previous = rating.value
  try {
    await api<void>(
      `/api/sessions/${props.sessionId}/messages/${props.messageId}/feedback`,
      { method: 'DELETE' },
    )
    rating.value = null
    savedReason.value = null
    savedReasonText.value = ''
    if (previous === 'like') {
      likes.value = Math.max(0, likes.value - 1)
    } else if (previous === 'dislike') {
      dislikes.value = Math.max(0, dislikes.value - 1)
    }
    panelOpen.value = false
    selectedReason.value = ''
    otherText.value = ''
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function onLike(): void {
  if (rating.value === 'like') {
    void cancel()
    return
  }
  void submitRating('like')
}

function openReasons(): void {
  selectedReason.value = savedReason.value ?? ''
  otherText.value = savedReasonText.value
  panelOpen.value = true
  error.value = ''
}

function onDislike(): void {
  if (rating.value === 'dislike') {
    void cancel()
    return
  }
  openReasons()
}

function submitDislike(): void {
  if (!canSubmitReason.value) {
    return
  }
  const text = selectedReason.value === 'other' ? otherText.value.trim() : null
  void submitRating('dislike', selectedReason.value, text)
}
</script>

<template>
  <div class="feedback-bar">
    <div class="feedback-actions">
      <button
        class="feedback-btn"
        :class="{ active: rating === 'like' }"
        type="button"
        :disabled="loading"
        title="有帮助"
        @click="onLike"
      >
        <span class="feedback-glyph">赞</span>
        <span class="feedback-num">{{ likes }}</span>
      </button>
      <button
        class="feedback-btn"
        :class="{ active: rating === 'dislike' }"
        type="button"
        :disabled="loading"
        title="没帮助"
        @click="onDislike"
      >
        <span class="feedback-glyph">踩</span>
        <span class="feedback-num">{{ dislikes }}</span>
      </button>
      <button v-if="rating" class="feedback-cancel" type="button" :disabled="loading" @click="cancel">
        取消评价
      </button>
      <button
        v-if="rating === 'dislike' && savedReason && !panelOpen"
        class="feedback-cancel"
        type="button"
        @click="openReasons"
      >
        修改原因
      </button>
      <span v-if="rating === 'dislike' && savedReason" class="feedback-reason-note">
        已选：{{ reasonLabel(savedReason) }}
        <template v-if="savedReason === 'other' && savedReasonText"> · {{ savedReasonText }}</template>
      </span>
    </div>

    <div v-if="panelOpen" class="feedback-panel">
      <div class="feedback-panel-title">请选择原因</div>
      <label v-for="reason in REASONS" :key="reason.code" class="feedback-reason">
        <input
          v-model="selectedReason"
          type="radio"
          :name="`reason-${messageId}`"
          :value="reason.code"
        />
        <span>{{ reason.label }}</span>
      </label>
      <textarea
        v-if="selectedReason === 'other'"
        v-model="otherText"
        class="feedback-other"
        :maxlength="reasonMax"
        rows="2"
        :placeholder="`请填写具体原因（1-${reasonMax} 字）`"
      ></textarea>
      <div class="feedback-panel-actions">
        <span v-if="selectedReason === 'other'" class="feedback-counter">
          {{ otherText.trim().length }}/{{ reasonMax }}
        </span>
        <button class="ghost-btn" type="button" @click="panelOpen = false">取消</button>
        <button
          class="primary-btn"
          type="button"
          :disabled="!canSubmitReason || loading"
          @click="submitDislike"
        >
          提交
        </button>
      </div>
    </div>

    <div v-if="error" class="feedback-error">{{ error }}</div>
  </div>
</template>
