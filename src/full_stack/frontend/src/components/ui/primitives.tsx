/**
 * Interface primitives.
 *
 * Every screen composes from these so spacing, focus behaviour, and the
 * progressive-disclosure pattern stay identical across the app.
 */

import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, ChevronRight, Loader2, X } from 'lucide-react'
import {
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from 'react'
import { createPortal } from 'react-dom'

/* --- Button --------------------------------------------------------------- */

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: 'sm' | 'md' | 'lg'
  icon?: ReactNode
  iconOnly?: boolean
  block?: boolean
  loading?: boolean
}

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  iconOnly,
  block,
  loading,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        'btn',
        `btn--${variant}`,
        size !== 'md' && `btn--${size}`,
        iconOnly && 'btn--icon',
        block && 'btn--block',
        className,
      )}
    >
      {loading ? <Loader2 size={14} className="spin" /> : icon}
      {!iconOnly && children}
    </button>
  )
}

/* --- Field ---------------------------------------------------------------- */

export interface FieldProps {
  label?: ReactNode
  hint?: ReactNode
  error?: ReactNode
  info?: ReactNode
  required?: boolean
  htmlFor?: string
  children: ReactNode
  className?: string
}

export function Field({ label, hint, error, info, required, htmlFor, children, className }: FieldProps) {
  return (
    <div className={clsx('field', className)}>
      {label && (
        <label className="field__label" htmlFor={htmlFor}>
          <span>
            {label}
            {required && <span style={{ color: 'var(--critical)' }}> *</span>}
          </span>
          {info && <InfoDot>{info}</InfoDot>}
        </label>
      )}
      {children}
      {error ? <span className="field__error">{error}</span> : hint ? <span className="field__hint">{hint}</span> : null}
    </div>
  )
}

export const Input = (props: InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean; mono?: boolean }) => {
  const { invalid, mono, className, ...rest } = props
  return <input {...rest} className={clsx('input', mono && 'input--mono', invalid && 'input--invalid', className)} />
}

export const Textarea = (props: TextareaHTMLAttributes<HTMLTextAreaElement> & { mono?: boolean }) => {
  const { mono, className, ...rest } = props
  return <textarea {...rest} className={clsx('textarea', mono && 'input--mono', className)} />
}

export const Select = (props: SelectHTMLAttributes<HTMLSelectElement>) => {
  const { className, ...rest } = props
  return <select {...rest} className={clsx('select', className)} />
}

/* --- Toggle and segmented ------------------------------------------------- */

export function Toggle({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
  label?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      className="toggle"
      onClick={() => onChange(!checked)}
    />
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = 'md',
  block = false,
}: {
  value: T
  options: { value: T; label: ReactNode; title?: string }[]
  onChange: (next: T) => void
  size?: 'sm' | 'md'
  /** Stretch to the container. Off by default so a two-option control stays small. */
  block?: boolean
}) {
  return (
    <div
      className={clsx('segmented', block && 'segmented--block')}
      role="group"
      style={size === 'sm' ? { padding: 2 } : undefined}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.title}
          aria-pressed={value === option.value}
          className="segmented__item"
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

/* --- Slider with a numeric companion -------------------------------------- */

/**
 * Range with a numeric companion.
 *
 * The track itself communicates the position, so the only labels are the two
 * bounds: preset chips underneath duplicated what the thumb already showed.
 */
export function SliderField({
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  suffix,
  disabled,
}: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (next: number) => void
  format?: (value: number) => string
  suffix?: string
  disabled?: boolean
}) {
  const label = (bound: number) => (format ? format(bound) : String(bound))
  return (
    <div className="stack gap-1">
      <div className="row gap-3">
        {/* The bounds belong under the track, so the track and the row that
            labels it are one column and the numeric companion sits outside
            it. Laying the labels out against the whole row instead pushed the
            upper bound past the track and under the number box. */}
        <div className="stack gap-1 grow" style={{ minWidth: 0 }}>
          <input
            type="range"
            className="slider"
            min={min}
            max={max}
            step={step}
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(Number(event.target.value))}
          />
          <div className="row between">
            <span className="t-micro faint tabular">{label(min)}</span>
            <span className="t-micro faint tabular">{label(max)}</span>
          </div>
        </div>
        <input
          type="number"
          className="input tabular"
          style={{ width: 88, flex: 'none' }}
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(event) => {
            const next = Number(event.target.value)
            if (!Number.isNaN(next)) onChange(Math.min(max, Math.max(min, next)))
          }}
        />
        {suffix && <span className="t-tiny muted" style={{ flex: 'none' }}>{suffix}</span>}
      </div>
    </div>
  )
}

/* --- Badge, chip, callout ------------------------------------------------- */

export function Badge({
  tone = 'neutral',
  mono,
  outline,
  children,
  className,
  style,
}: {
  tone?: 'neutral' | 'accent' | 'positive' | 'caution' | 'critical' | 'info'
  mono?: boolean
  outline?: boolean
  children: ReactNode
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <span
      className={clsx(
        'badge',
        tone !== 'neutral' && `badge--${tone}`,
        mono && 'badge--mono',
        outline && 'badge--outline',
        className,
      )}
      style={style}
    >
      {children}
    </span>
  )
}

export function Callout({
  tone = 'neutral',
  icon,
  title,
  children,
  action,
}: {
  tone?: 'neutral' | 'info' | 'caution' | 'critical' | 'positive'
  icon?: ReactNode
  title?: ReactNode
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className={clsx('callout', `callout--${tone}`)}>
      {icon && <span className="callout__icon">{icon}</span>}
      <div className="grow stack gap-1">
        {title && <span className="semibold">{title}</span>}
        {children && <div className="secondary">{children}</div>}
      </div>
      {action && <div style={{ flex: 'none' }}>{action}</div>}
    </div>
  )
}

/* --- Tooltip -------------------------------------------------------------- */

export function Tooltip({ content, children, side = 'top' }: { content: ReactNode; children: ReactNode; side?: 'top' | 'bottom' | 'right' }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const ref = useRef<HTMLSpanElement>(null)

  const show = () => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    if (side === 'right') setPos({ x: rect.right + 8, y: rect.top + rect.height / 2 })
    else if (side === 'bottom') setPos({ x: rect.left + rect.width / 2, y: rect.bottom + 8 })
    else setPos({ x: rect.left + rect.width / 2, y: rect.top - 8 })
    setOpen(true)
  }

  if (!content) return <>{children}</>

  return (
    <>
      <span
        ref={ref}
        onMouseEnter={show}
        onMouseLeave={() => setOpen(false)}
        onFocus={show}
        onBlur={() => setOpen(false)}
        style={{ display: 'inline-flex', minWidth: 0 }}
      >
        {children}
      </span>
      {open &&
        createPortal(
          <div
            className="tip"
            style={{
              left: pos.x,
              top: pos.y,
              transform:
                side === 'right'
                  ? 'translateY(-50%)'
                  : side === 'bottom'
                    ? 'translateX(-50%)'
                    : 'translate(-50%, -100%)',
            }}
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  )
}

/* --- Info disclosure ------------------------------------------------------ */

/**
 * The `i` control. Detail stays collapsed until asked for, which is how the
 * interface stays calm while still carrying everything a newcomer needs.
 */
export function InfoDot({ children, label = 'More information' }: { children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <>
      <button
        type="button"
        className="infodot"
        aria-expanded={open}
        aria-controls={id}
        aria-label={label}
        title={label}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          setOpen((v) => !v)
        }}
      >
        i
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            id={id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            style={{ overflow: 'hidden', width: '100%', order: 99, flexBasis: '100%' }}
          >
            <div className="explainer">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

/* --- Disclosure ----------------------------------------------------------- */

export function Disclosure({
  title,
  subtitle,
  defaultOpen = false,
  open: controlled,
  onOpenChange,
  right,
  children,
}: {
  title: ReactNode
  subtitle?: ReactNode
  defaultOpen?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  right?: ReactNode
  children: ReactNode
}) {
  const [internal, setInternal] = useState(defaultOpen)
  const open = controlled ?? internal
  const setOpen = (next: boolean) => {
    setInternal(next)
    onOpenChange?.(next)
  }
  return (
    <div className="disclosure" data-open={open}>
      <button type="button" className="disclosure__trigger" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronRight size={15} className="disclosure__chevron" />
        <span className="grow stack" style={{ gap: 1 }}>
          <span className="semibold t-small">{title}</span>
          {subtitle && <span className="t-tiny muted">{subtitle}</span>}
        </span>
        {right}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <div className="disclosure__panel" style={{ paddingTop: 'var(--s-4)' }}>
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* --- Card ----------------------------------------------------------------- */

export function Card({
  title,
  subtitle,
  actions,
  children,
  flush,
  raised,
  className,
  style,
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  flush?: boolean
  raised?: boolean
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <section className={clsx('card', raised && 'card--raised', className)} style={style}>
      {(title || actions) && (
        <header className="card__header">
          <div className="stack" style={{ gap: 1, minWidth: 0 }}>
            {title && <span className="card__title truncate">{title}</span>}
            {subtitle && <span className="t-tiny muted">{subtitle}</span>}
          </div>
          {actions && <div className="row gap-2" style={{ flex: 'none' }}>{actions}</div>}
        </header>
      )}
      <div className={clsx('card__body', flush && 'card__body--flush')}>{children}</div>
    </section>
  )
}

/* --- Tabs ----------------------------------------------------------------- */

export function Tabs<T extends string>({
  value,
  options,
  onChange,
  spread = false,
}: {
  value: T
  options: { value: T; label: ReactNode; badge?: ReactNode; disabled?: boolean }[]
  onChange: (next: T) => void
  /** Share the full width evenly rather than bunching every label on the left. */
  spread?: boolean
}) {
  return (
    <div className={clsx('tabs', spread && 'tabs--spread')} role="tablist">
      {options.map((option) => (
        <button
          key={option.value}
          role="tab"
          type="button"
          disabled={option.disabled}
          aria-selected={value === option.value}
          className="tabs__item"
          onClick={() => onChange(option.value)}
          style={option.disabled ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
        >
          {option.label}
          {option.badge}
        </button>
      ))}
    </div>
  )
}

/* --- Modal and drawer ----------------------------------------------------- */

function useEscape(onClose: () => void, active: boolean) {
  useEffect(() => {
    if (!active) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, active])
}

export function Modal({
  open,
  onClose,
  title,
  subtitle,
  footer,
  width,
  children,
}: {
  open: boolean
  onClose: () => void
  title?: ReactNode
  subtitle?: ReactNode
  footer?: ReactNode
  width?: number
  children: ReactNode
}) {
  useEscape(onClose, open)
  return createPortal(
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="scrim"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
          <motion.div
            className="modal"
            role="dialog"
            aria-modal="true"
            style={width ? { width: `min(96vw, ${width}px)` } : undefined}
            initial={{ opacity: 0, scale: 0.97, y: '-46%', x: '-50%' }}
            animate={{ opacity: 1, scale: 1, y: '-50%', x: '-50%' }}
            exit={{ opacity: 0, scale: 0.98, y: '-48%', x: '-50%' }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            {(title || subtitle) && (
              <header className="sheet__header">
                <div className="stack" style={{ gap: 2 }}>
                  {title && <span className="t-h3">{title}</span>}
                  {subtitle && <span className="t-small muted">{subtitle}</span>}
                </div>
                <Button variant="ghost" size="sm" iconOnly icon={<X size={15} />} onClick={onClose} aria-label="Close" />
              </header>
            )}
            <div className="sheet__body">{children}</div>
            {footer && <footer className="sheet__footer">{footer}</footer>}
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body,
  )
}

export function Drawer({
  open,
  onClose,
  title,
  subtitle,
  footer,
  width,
  children,
}: {
  open: boolean
  onClose: () => void
  title?: ReactNode
  subtitle?: ReactNode
  footer?: ReactNode
  width?: number
  children: ReactNode
}) {
  useEscape(onClose, open)
  return createPortal(
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="scrim" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
          <motion.aside
            className="drawer"
            role="dialog"
            aria-modal="true"
            style={width ? { width: `min(96vw, ${width}px)` } : undefined}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <header className="sheet__header">
              <div className="stack" style={{ gap: 2, minWidth: 0 }}>
                {title && <span className="t-h3 truncate">{title}</span>}
                {subtitle && <span className="t-small muted truncate">{subtitle}</span>}
              </div>
              <Button variant="ghost" size="sm" iconOnly icon={<X size={15} />} onClick={onClose} aria-label="Close" />
            </header>
            <div className="sheet__body">{children}</div>
            {footer && <footer className="sheet__footer">{footer}</footer>}
          </motion.aside>
        </>
      )}
    </AnimatePresence>,
    document.body,
  )
}

/* --- Feedback ------------------------------------------------------------- */

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: ReactNode
  title: ReactNode
  body?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty">
      {icon && <div className="empty__icon">{icon}</div>}
      <div className="stack gap-1" style={{ maxWidth: 420 }}>
        <span className="t-body semibold" style={{ color: 'var(--text)' }}>
          {title}
        </span>
        {body && <span className="t-small">{body}</span>}
      </div>
      {action}
    </div>
  )
}

export function Skeleton({ height = 14, width, radius }: { height?: number; width?: number | string; radius?: number }) {
  return <div className="skeleton" style={{ height, width: width ?? '100%', borderRadius: radius }} />
}

export function Spinner({ size = 15 }: { size?: number }) {
  return <Loader2 size={size} className="spin" style={{ color: 'var(--text-muted)' }} />
}

export function Progress({ value, max = 1, indeterminate }: { value?: number; max?: number; indeterminate?: boolean }) {
  const pct = indeterminate ? 0 : Math.max(0, Math.min(1, (value ?? 0) / (max || 1))) * 100
  return (
    <div className={clsx('progress', indeterminate && 'progress--indeterminate')}>
      <div className="progress__fill" style={indeterminate ? undefined : { width: `${pct}%` }} />
    </div>
  )
}

/* --- Copy ----------------------------------------------------------------- */

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <Button
      variant="ghost"
      size="sm"
      icon={done ? <Check size={13} /> : undefined}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setDone(true)
          window.setTimeout(() => setDone(false), 1600)
        } catch {
          /* clipboard access can be denied; the button simply does nothing */
        }
      }}
    >
      {done ? 'Copied' : label}
    </Button>
  )
}

/* --- Scroll shadow context (used by long panels) -------------------------- */

const ScrollContext = createContext<HTMLElement | null>(null)
export const useScrollParent = () => useContext(ScrollContext)

export function ScrollArea({ children, className, style }: { children: ReactNode; className?: string; style?: React.CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null)
  const [element, setElement] = useState<HTMLElement | null>(null)
  useLayoutEffect(() => setElement(ref.current), [])
  return (
    <ScrollContext.Provider value={element}>
      <div ref={ref} className={className} style={{ overflowY: 'auto', ...style }}>
        {children}
      </div>
    </ScrollContext.Provider>
  )
}
