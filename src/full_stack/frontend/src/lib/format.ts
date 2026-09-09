/** Presentation helpers. Formatting lives in one place so numbers read alike. */

let locale = 'en-US'
export const setLocale = (next: string) => {
  locale = next || 'en-US'
}

export function compactNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  const abs = Math.abs(value)
  if (abs < 1000) return String(Math.round(value))
  return new Intl.NumberFormat(locale, { notation: 'compact', maximumFractionDigits: digits }).format(value)
}

export function number(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return new Intl.NumberFormat(locale, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)
}

export function tokens(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return new Intl.NumberFormat(locale).format(Math.round(value))
}

/** Small amounts need more decimals than large ones to stay meaningful. */
export function usd(value: number | null | undefined, force?: number): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  const digits = force ?? (Math.abs(value) >= 1 ? 2 : Math.abs(value) >= 0.01 ? 3 : 4)
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

export function usdPerMillion(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'n/a'
  if (value === 0) return 'free'
  const digits = value >= 1 ? 2 : value >= 0.1 ? 3 : 4
  return `$${value.toFixed(digits)}/M`
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return `${(value * 100).toFixed(digits)}%`
}

export function signed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  const text = value.toFixed(digits)
  return value > 0 ? `+${text}` : text
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '-'
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, '0')}s`
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

export function elapsedSince(iso: string | null | undefined, until?: string | null): number | null {
  if (!iso) return null
  const start = Date.parse(iso)
  if (Number.isNaN(start)) return null
  const end = until ? Date.parse(until) : Date.now()
  return (Number.isNaN(end) ? Date.now() : end) - start > 0 ? ((Number.isNaN(end) ? Date.now() : end) - start) / 1000 : 0
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '-'
  const then = Date.parse(iso)
  if (Number.isNaN(then)) return '-'
  const seconds = Math.round((Date.now() - then) / 1000)
  if (seconds < 45) return 'just now'
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })
  const table: [Intl.RelativeTimeFormatUnit, number][] = [
    ['second', 60],
    ['minute', 60],
    ['hour', 24],
    ['day', 7],
    ['week', 4.35],
    ['month', 12],
  ]
  let value = seconds
  for (const [unit, span] of table) {
    if (Math.abs(value) < span) return formatter.format(-Math.round(value), unit)
    value /= span
  }
  return formatter.format(-Math.round(value), 'year')
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return '-'
  const parsed = Date.parse(iso)
  if (Number.isNaN(parsed)) return '-'
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' }).format(parsed)
}

export function bytes(value: number | null | undefined): string {
  if (!value) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

/** Machine ids repeat their ancestry; the tail is the informative part. */
export function humaniseId(value: string): string {
  const text = String(value ?? '').trim()
  if (!text) return ''
  const tail = text.includes('__') ? text.split('__').pop()! : text
  const spaced = tail.replace(/_/g, ' ').trim()
  if (!spaced) return text
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export function titleCase(value: string): string {
  return String(value ?? '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function modelShortName(id: string): string {
  const [, name] = String(id ?? '').split('/')
  return name ?? id
}

export function providerOf(id: string): string {
  const parts = String(id ?? '').split('/')
  return parts.length > 1 ? parts[0] : 'other'
}
