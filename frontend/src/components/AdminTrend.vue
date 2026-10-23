<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { TrendPoint } from '@/types'

echarts.use([BarChart, LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const props = defineProps<{
  points: TrendPoint[]
}>()

const container = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

function percent(value: number | null): number | null {
  return value === null ? null : Math.round(value * 1000) / 10
}

function render(): void {
  if (!container.value) {
    return
  }
  if (chart && chart.getDom() !== container.value) {
    chart.dispose()
    chart = null
  }
  if (!chart) {
    chart = echarts.init(container.value)
  }
  const labels = props.points.map((item) => item.label)
  chart.setOption(
    {
      animation: false,
      grid: { left: 48, right: 56, top: 48, bottom: 34 },
      tooltip: { trigger: 'axis' },
      legend: { top: 8, textStyle: { color: '#5b6270', fontSize: 11 } },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: '#9096a3', fontSize: 10 },
        axisLine: { lineStyle: { color: '#e6e8ef' } },
      },
      yAxis: [
        {
          type: 'value',
          name: '轮次 / %',
          axisLabel: { color: '#9096a3', fontSize: 10 },
          splitLine: { lineStyle: { color: '#f0f1f6' } },
        },
        {
          type: 'value',
          name: 'P95 ms',
          axisLabel: { color: '#9096a3', fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: '轮次',
          type: 'bar',
          data: props.points.map((item) => item.turns),
          itemStyle: { color: '#b9c0f5', borderRadius: [3, 3, 0, 0] },
          barMaxWidth: 18,
        },
        {
          name: '工具成功率',
          type: 'line',
          smooth: true,
          data: props.points.map((item) => percent(item.tool_success_rate)),
          itemStyle: { color: '#4f5bd5' },
        },
        {
          name: '满意率',
          type: 'line',
          smooth: true,
          data: props.points.map((item) => percent(item.satisfaction_rate)),
          itemStyle: { color: '#0f766e' },
        },
        {
          name: 'P95 耗时',
          type: 'line',
          smooth: true,
          yAxisIndex: 1,
          data: props.points.map((item) => item.p95_ms),
          itemStyle: { color: '#b45309' },
        },
      ],
    },
    true,
  )
}

function resize(): void {
  chart?.resize()
}

onMounted(async () => {
  await nextTick()
  render()
  window.addEventListener('resize', resize)
})

watch(
  () => props.points,
  async () => {
    if (!props.points.length) {
      chart?.dispose()
      chart = null
      return
    }
    await nextTick()
    render()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div v-if="points.length" ref="container" class="admin-chart"></div>
  <div v-else class="admin-empty">所选范围内暂无趋势数据</div>
</template>
