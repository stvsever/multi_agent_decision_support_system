/** Small presentational pieces shared by the report tabs. */

import type { ReactNode } from 'react'
import { Check, Minus, X } from 'lucide-react'
import { percent } from '@/lib/format'

export function Stat({ label, value, meta }: { label: ReactNode; value: ReactNode; meta?: ReactNode }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      {meta ? <span className="stat__meta">{meta}</span> : null}
    </div>
  )
}

export function GroupBox({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="reports__group-box">
      <span className="reports__group-title">{title}</span>
      {children}
    </div>
  )
}

export function KeyValue({ rows }: { rows: { label: ReactNode; value: ReactNode }[] }) {
  if (rows.length === 0) return null
  return (
    <dl className="kv">
      {rows.map((row, index) => (
        <div key={index} style={{ display: 'contents' }}>
          <dt>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** A boolean rendered as a pass or fail mark, with the unknown case kept distinct. */
export function Flag({ label, state }: { label: ReactNode; state: boolean | null }) {
  const className =
    state === true ? 'reports__flag reports__flag--pass' : state === false ? 'reports__flag reports__flag--fail' : 'reports__flag'
  return (
    <span className={className}>
      {state === true ? <Check size={11} /> : state === false ? <X size={11} /> : <Minus size={11} />}
      {label}
    </span>
  )
}

export function Bar({
  label,
  value,
  lead,
  caption,
}: {
  label: ReactNode
  value: number
  lead?: boolean
  caption?: string
}) {
  const width = Math.max(0, Math.min(1, value)) * 100
  // Inside a table cell the bar carries its own column header, so the label
  // column is dropped rather than left empty.
  const labelled = label !== '' && label !== null && label !== undefined
  return (
    <div className="reports__bar-row" style={labelled ? undefined : { gridTemplateColumns: '1fr 56px' }}>
      {labelled && (
        <span className="truncate" title={typeof label === 'string' ? label : undefined}>
          {label}
        </span>
      )}
      <span className="reports__bar-track">
        <span className={lead ? 'reports__bar-fill reports__bar-fill--lead' : 'reports__bar-fill'} style={{ width: `${width}%` }} />
      </span>
      <span className="reports__bar-value">{caption ?? percent(value, 1)}</span>
    </div>
  )
}

export function BulletList({ items, ordered }: { items: string[]; ordered?: boolean }) {
  if (items.length === 0) return null
  const className = ordered ? 'md__list md__list--ordered' : 'md__list md__list--bullet'
  return ordered ? (
    <ol className={className} style={{ fontSize: 'var(--t-small)', color: 'var(--text-secondary)' }}>
      {items.map((item, index) => (
        <li key={index} className="md__li">
          {item}
        </li>
      ))}
    </ol>
  ) : (
    <ul className={className} style={{ fontSize: 'var(--t-small)', color: 'var(--text-secondary)' }}>
      {items.map((item, index) => (
        <li key={index} className="md__li">
          {item}
        </li>
      ))}
    </ul>
  )
}

export function NotRecorded({ what }: { what: string }) {
  return <span className="t-small muted">{what} was not recorded in this run.</span>
}
