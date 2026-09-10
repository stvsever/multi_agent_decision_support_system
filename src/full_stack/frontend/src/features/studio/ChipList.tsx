/** A list of short labels, typed one at a time. Shared by the flat form and the tree. */

import clsx from 'clsx'
import { X } from 'lucide-react'
import { useCallback, type AriaAttributes } from 'react'

/** The entry field is the control, so the ARIA a caller sets lands on it. */
type ChipListAria = Pick<AriaAttributes, 'aria-invalid' | 'aria-describedby' | 'aria-label'>

export function ChipList({
  values,
  onChange,
  placeholder,
  invalid,
  onBlurCapture,
  ...aria
}: {
  values: string[]
  onChange: (next: string[]) => void
  placeholder: string
  invalid?: boolean
  onBlurCapture?: () => void
} & ChipListAria) {
  const add = useCallback(
    (raw: string) => {
      const parts = raw
        .split(',')
        .map((part) => part.trim())
        .filter(Boolean)
      if (!parts.length) return
      const next = [...values]
      for (const part of parts) if (!next.includes(part)) next.push(part)
      onChange(next)
    },
    [values, onChange],
  )

  return (
    <div className={clsx('chiplist', invalid && 'chiplist--invalid')}>
      {values.map((value, index) => (
        <span key={`${value}-${index}`} className="chiplist__chip">
          <span className="mono truncate">{value}</span>
          <button
            type="button"
            aria-label={`Remove ${value}`}
            onClick={() => onChange(values.filter((_, position) => position !== index))}
          >
            <X size={11} />
          </button>
        </span>
      ))}
      <input
        className="chiplist__input"
        aria-invalid={invalid || undefined}
        {...aria}
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ',') {
            event.preventDefault()
            add(event.currentTarget.value)
            event.currentTarget.value = ''
          }
          if (event.key === 'Backspace' && !event.currentTarget.value && values.length) {
            onChange(values.slice(0, -1))
          }
        }}
        onBlur={(event) => {
          add(event.currentTarget.value)
          event.currentTarget.value = ''
          onBlurCapture?.()
        }}
      />
    </div>
  )
}
