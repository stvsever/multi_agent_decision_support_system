/** Small pieces the four views and both rails share. */

import type { CSSProperties, ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import type { Band } from '@/lib/types'
import { signed } from '@/lib/format'
import { BAND_LABEL, bandColor, bandFor, prettyLabel, type FlatNode } from './model'

export function BandDot({ band, size = 8 }: { band: Band; size?: number }) {
  return (
    <span
      className="onto-dot"
      aria-hidden="true"
      style={{ width: size, height: size, background: bandColor(band) }}
    />
  )
}

/**
 * The signed value for a measurement, the subtree mean magnitude for a branch.
 * Both print in full even when the bar beside them is clamped.
 */
export function ScoreChip({
  band,
  value,
  aggregate,
  title,
}: {
  band: Band
  value: number | null
  aggregate?: boolean
  title?: string
}) {
  const text =
    value === null || !Number.isFinite(value) ? '-' : aggregate ? value.toFixed(2) : signed(value)
  return (
    <span
      className="onto-chip tabular"
      data-aggregate={aggregate || undefined}
      style={{ '--tone': bandColor(band) } as CSSProperties}
      title={title ?? `${BAND_LABEL[band]}${value === null ? '' : `, ${text}`}`}
    >
      {text}
    </span>
  )
}

/** Diverging bar centred on zero. The scale is clamped; the number is not. */
export function DeviationBar({
  score,
  clamp,
  width = 84,
  height = 8,
}: {
  score: number | null
  clamp: number
  width?: number
  height?: number
}) {
  const usable = width / 2 - 1
  const magnitude =
    score === null || !Number.isFinite(score) ? 0 : Math.min(Math.abs(score) / clamp, 1) * usable
  const overflow = score !== null && Number.isFinite(score) && Math.abs(score) > clamp
  const sign = score === null || !Number.isFinite(score) ? 'none' : score < 0 ? 'neg' : 'pos'
  return (
    <span
      className="onto-bar"
      aria-hidden="true"
      data-sign={sign}
      data-over={overflow || undefined}
      style={
        {
          width,
          height,
          '--fill': `${Math.max(magnitude, score === null ? 0 : 1.5)}px`,
          '--tone': bandColor(score === null ? 'missing' : bandFor(score)),
        } as CSSProperties
      }
    >
      <span className="onto-bar__fill" />
      <span className="onto-bar__zero" />
    </span>
  )
}

/** Single direction magnitude bar, used for coverage and child rankings. */
export function MagnitudeBar({
  value,
  max,
  band,
  height = 6,
}: {
  value: number
  max: number
  band: Band
  height?: number
}) {
  const fraction = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0
  return (
    <span className="onto-mag" style={{ height }} aria-hidden="true">
      <span
        className="onto-mag__fill"
        style={{ width: `${fraction * 100}%`, background: bandColor(band) }}
      />
    </span>
  )
}

export function Highlight({ text, query }: { text: string; query: string }) {
  const needle = query.trim().toLowerCase()
  if (!needle) return <>{text}</>
  const lower = text.toLowerCase()
  const parts: ReactNode[] = []
  let cursor = 0
  let at = lower.indexOf(needle)
  if (at < 0) return <>{text}</>
  while (at >= 0) {
    if (at > cursor) parts.push(text.slice(cursor, at))
    parts.push(
      <mark key={`${at}`} className="onto-mark">
        {text.slice(at, at + needle.length)}
      </mark>,
    )
    cursor = at + needle.length
    at = lower.indexOf(needle, cursor)
  }
  if (cursor < text.length) parts.push(text.slice(cursor))
  return <>{parts}</>
}

export function Breadcrumb({
  chain,
  onSelect,
  compact,
}: {
  chain: { key: string; label: string }[]
  onSelect?: (key: string) => void
  compact?: boolean
}) {
  if (chain.length === 0) return null
  return (
    <nav className="onto-crumbs" data-compact={compact || undefined} aria-label="Ontology path">
      {chain.map((step, index) => (
        <span className="onto-crumbs__step" key={step.key}>
          {index > 0 && <ChevronRight size={11} className="onto-crumbs__sep" aria-hidden="true" />}
          {onSelect ? (
            <button type="button" className="onto-crumbs__link" onClick={() => onSelect(step.key)}>
              {step.label}
            </button>
          ) : (
            <span className="onto-crumbs__link">{step.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}

export const crumbsOf = (chain: FlatNode[]): { key: string; label: string }[] =>
  chain.map((flat) => ({ key: flat.key, label: prettyLabel(flat.node.label) }))

/** Cursor following tooltip for the SVG views, where hover targets are paths. */
export function ChartTooltip({
  x,
  y,
  width,
  children,
}: {
  x: number
  y: number
  width: number
  children: ReactNode
}) {
  const flip = x > width - 220
  return (
    <div
      className="onto-tip"
      role="presentation"
      style={{ left: x, top: y, transform: `translate(${flip ? '-100%' : '0'}, -100%)` }}
    >
      {children}
    </div>
  )
}

export function Stat({ label, value, meta }: { label: ReactNode; value: ReactNode; meta?: ReactNode }) {
  return (
    <div className="stat onto-stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      {meta && <span className="stat__meta">{meta}</span>}
    </div>
  )
}
