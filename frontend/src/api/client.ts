export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`
    try {
      const text = await response.text()
      if (text) {
        try {
          const parsed = JSON.parse(text) as { detail?: unknown }
          detail =
            typeof parsed.detail === 'string' ? parsed.detail : text.slice(0, 300)
        } catch {
          detail = text.slice(0, 300)
        }
      }
    } catch {
      // keep default detail
    }
    throw new Error(detail)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export interface AppConfig {
  citation_display_limit: number
  citation_max_total: number
  feedback_reason_max: number
  admin_page_size: number
}

let configPromise: Promise<AppConfig> | null = null

export function loadAppConfig(): Promise<AppConfig> {
  if (!configPromise) {
    configPromise = api<AppConfig>('/api/config').catch(() => ({
      citation_display_limit: 3,
      citation_max_total: 10,
      feedback_reason_max: 200,
      admin_page_size: 50,
    }))
  }
  return configPromise
}

export function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

export function fmtPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '—'
  }
  return `${(value * 100).toFixed(1)}%`
}

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '—'
  }
  return `${Math.round(value)} ms`
}

export function fmtDateTime(value: string | null | undefined): string {
  if (!value) {
    return '—'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })
}
