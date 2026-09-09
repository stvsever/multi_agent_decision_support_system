/**
 * Settings controls.
 *
 * Typed and dragged values keep a local draft and settle after a pause; every
 * other control commits on the interaction itself. Numeric drafts are clamped
 * to the range the service declares, so a value that cannot be saved never
 * leaves the field.
 */

import { Check, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Field, InfoDot, Input, SliderField, Spinner, Textarea, Toggle } from '@/components/ui/primitives'
import { useDebounced } from '@/lib/hooks'
import { useSettingsController, type ConfigSection, type SaveStatus } from './state'

const SETTLE_MS = 400

/* --- Section scaffolding -------------------------------------------------- */

export function SectionHead({
  title,
  description,
  info,
}: {
  title: string
  description: ReactNode
  info?: ReactNode
}) {
  return (
    <header className="settings__head wrap">
      <div className="stack grow" style={{ gap: 2, minWidth: 0 }}>
        <span className="t-h3">{title}</span>
        <span className="t-small muted">{description}</span>
      </div>
      {info && <InfoDot label={`About ${title.toLowerCase()}`}>{info}</InfoDot>}
    </header>
  )
}

export function Group({ title, children }: { title?: ReactNode; children: ReactNode }) {
  return (
    <div className="settings__group">
      {title && <span className="eyebrow">{title}</span>}
      {children}
    </div>
  )
}

export function Grid({ children, columns = 2 }: { children: ReactNode; columns?: 2 | 3 }) {
  return <div className={columns === 3 ? 'settings__grid settings__grid--three' : 'settings__grid'}>{children}</div>
}

/* --- Text ----------------------------------------------------------------- */

export function TextControl({
  value,
  onCommit,
  placeholder,
  mono,
  type = 'text',
  id,
  ariaLabel,
}: {
  value: string
  onCommit: (next: string) => void
  placeholder?: string
  mono?: boolean
  type?: string
  id?: string
  ariaLabel?: string
}) {
  const [draft, setDraft] = useState(value)
  const editing = useRef(false)
  const settled = useDebounced(draft, SETTLE_MS)

  useEffect(() => {
    if (!editing.current) setDraft(value)
  }, [value])

  useEffect(() => {
    if (!editing.current || settled === value) return
    onCommit(settled)
    // Only the settled draft should trigger a save.
  }, [settled])

  return (
    <Input
      id={id}
      aria-label={ariaLabel}
      type={type}
      mono={mono}
      value={draft}
      placeholder={placeholder}
      onChange={(event) => {
        editing.current = true
        setDraft(event.target.value)
      }}
      onBlur={() => {
        editing.current = false
        if (draft !== value) onCommit(draft)
      }}
    />
  )
}

export function TextareaControl({
  value,
  onCommit,
  placeholder,
  rows = 5,
  mono,
  className,
  ariaLabel,
}: {
  value: string
  onCommit: (next: string) => void
  placeholder?: string
  rows?: number
  mono?: boolean
  className?: string
  ariaLabel?: string
}) {
  const [draft, setDraft] = useState(value)
  const editing = useRef(false)
  const settled = useDebounced(draft, SETTLE_MS)

  useEffect(() => {
    if (!editing.current) setDraft(value)
  }, [value])

  useEffect(() => {
    if (!editing.current || settled === value) return
    onCommit(settled)
  }, [settled])

  return (
    <div className="stack gap-1">
      <Textarea
        aria-label={ariaLabel}
        className={className}
        mono={mono}
        rows={rows}
        value={draft}
        placeholder={placeholder}
        onChange={(event) => {
          editing.current = true
          setDraft(event.target.value)
        }}
        onBlur={() => {
          editing.current = false
          if (draft !== value) onCommit(draft)
        }}
      />
      <span className="settings__count">{draft.length.toLocaleString()} characters</span>
    </div>
  )
}

/* --- Numbers -------------------------------------------------------------- */

export function NumberControl({
  value,
  onCommit,
  min,
  max,
  step = 1,
  integer = true,
  width,
  id,
  ariaLabel,
}: {
  value: number
  onCommit: (next: number) => void
  min: number
  max: number
  step?: number
  integer?: boolean
  width?: number
  id?: string
  ariaLabel?: string
}) {
  const [draft, setDraft] = useState(() => String(value))
  const editing = useRef(false)
  const settled = useDebounced(draft, SETTLE_MS)

  useEffect(() => {
    if (!editing.current) setDraft(String(value))
  }, [value])

  useEffect(() => {
    if (!editing.current) return
    const parsed = parse(settled, integer)
    // Mid-typing values outside the range are left alone; blur clamps them.
    if (parsed === null || parsed < min || parsed > max || parsed === value) return
    onCommit(parsed)
  }, [settled])

  const out = parse(draft, integer)
  const invalid = out !== null && (out < min || out > max)

  return (
    <Input
      id={id}
      aria-label={ariaLabel}
      className="tabular"
      type="number"
      inputMode={integer ? 'numeric' : 'decimal'}
      min={min}
      max={max}
      step={step}
      value={draft}
      invalid={invalid}
      style={width ? { width, flex: 'none' } : undefined}
      onChange={(event) => {
        editing.current = true
        setDraft(event.target.value)
      }}
      onBlur={() => {
        editing.current = false
        const parsed = parse(draft, integer)
        const next = parsed === null ? value : Math.min(max, Math.max(min, parsed))
        setDraft(String(next))
        if (next !== value) onCommit(next)
      }}
    />
  )
}

function parse(text: string, integer: boolean): number | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed)) return null
  return integer ? Math.round(parsed) : parsed
}

export function SliderControl({
  value,
  onCommit,
  min,
  max,
  step = 1,
  presets,
  suffix,
  format,
}: {
  value: number
  onCommit: (next: number) => void
  min: number
  max: number
  step?: number
  presets?: number[]
  suffix?: string
  format?: (value: number) => string
}) {
  const [draft, setDraft] = useState(value)
  const dragging = useRef(false)
  const settled = useDebounced(draft, SETTLE_MS)

  useEffect(() => {
    if (!dragging.current) setDraft(value)
  }, [value])

  useEffect(() => {
    if (!dragging.current || settled === value) return
    dragging.current = false
    onCommit(settled)
  }, [settled])

  return (
    <SliderField
      value={draft}
      min={min}
      max={max}
      step={step}
      presets={presets}
      suffix={suffix}
      format={format}
      onChange={(next) => {
        dragging.current = true
        setDraft(Math.min(max, Math.max(min, next)))
      }}
    />
  )
}

/* --- Switches ------------------------------------------------------------- */

export function SwitchRow({
  label,
  hint,
  info,
  checked,
  onChange,
  disabled,
}: {
  label: string
  hint?: ReactNode
  info?: ReactNode
  checked: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
}) {
  return (
    <div className="settings__switch">
      <span className="settings__switch-text">
        <span className="t-small semibold">{label}</span>
        {hint && <span className="t-tiny muted">{hint}</span>}
      </span>
      {info && <InfoDot label={`About ${label.toLowerCase()}`}>{info}</InfoDot>}
      <Toggle checked={checked} onChange={onChange} disabled={disabled} label={label} />
    </div>
  )
}

/* --- Bound helpers -------------------------------------------------------- */

/**
 * A number field bound to one key of one configuration section. Ranges come
 * from the service schema and are enforced before anything is sent.
 */
export function NumberSetting({
  label,
  hint,
  info,
  section,
  field,
  min,
  max,
  step,
  integer = true,
  suffix,
}: {
  label: string
  hint?: ReactNode
  info?: ReactNode
  section: ConfigSection
  field: string
  min: number
  max: number
  step?: number
  integer?: boolean
  suffix?: string
}) {
  const { config, update } = useSettingsController()
  const bucket = config[section] as unknown as Record<string, number>
  const value = Number(bucket[field] ?? 0)

  return (
    <Field label={label} hint={hint} info={info}>
      <div className="row gap-2">
        <NumberControl
          ariaLabel={label}
          value={value}
          min={min}
          max={max}
          step={step}
          integer={integer}
          onCommit={(next) => update(section, { [field]: next } as never)}
        />
        {suffix && (
          <span className="t-tiny muted" style={{ flex: 'none' }}>
            {suffix}
          </span>
        )}
      </div>
    </Field>
  )
}

export function TextSetting({
  label,
  hint,
  info,
  section,
  field,
  placeholder,
  mono,
}: {
  label: string
  hint?: ReactNode
  info?: ReactNode
  section: ConfigSection
  field: string
  placeholder?: string
  mono?: boolean
}) {
  const { config, update } = useSettingsController()
  const bucket = config[section] as unknown as Record<string, string>

  return (
    <Field label={label} hint={hint} info={info}>
      <TextControl
        ariaLabel={label}
        value={String(bucket[field] ?? '')}
        placeholder={placeholder}
        mono={mono}
        onCommit={(next) => update(section, { [field]: next } as never)}
      />
    </Field>
  )
}

/* --- Save indicator ------------------------------------------------------- */

export function SaveIndicator({ status }: { status: SaveStatus }) {
  if (status === 'idle') return null
  return (
    <div className="settings__saved" data-status={status} role="status" aria-live="polite">
      {status === 'saving' && (
        <>
          <Spinner size={12} /> Saving
        </>
      )}
      {status === 'saved' && (
        <>
          <Check size={12} /> Saved
        </>
      )}
      {status === 'error' && (
        <>
          <TriangleAlert size={12} /> Not saved
        </>
      )}
    </div>
  )
}
