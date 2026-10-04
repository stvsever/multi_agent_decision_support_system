/**
 * Optional measurement range for one regression output. A structured decision
 * model places its estimate on this range, so it should be the instrument's
 * real minimum and maximum; left empty, the companion LLM proposes one.
 */
import { Input } from '@/components/ui/primitives'
import type { OutputScale } from '@/lib/types'

function parse(text: string): number | undefined {
  if (text.trim() === '') return undefined
  const value = Number(text)
  return Number.isFinite(value) ? value : undefined
}

export function ScaleFields({
  output,
  scale,
  onChange,
}: {
  output: string
  scale: OutputScale | undefined
  onChange: (next: OutputScale | undefined) => void
}) {
  const update = (patch: Partial<OutputScale>) => {
    const next: OutputScale = { ...(scale ?? {}), ...patch }
    if (next.min === undefined) delete next.min
    if (next.max === undefined) delete next.max
    const integer = next.min !== undefined && next.max !== undefined && Number.isInteger(next.min) && Number.isInteger(next.max)
    if (next.min === undefined && next.max === undefined) onChange(undefined)
    else onChange({ ...next, integer })
  }
  return (
    <div className="row gap-2" style={{ flex: 'none' }}>
      <Input
        style={{ width: 82 }}
        inputMode="decimal"
        value={scale?.min ?? ''}
        placeholder="min"
        aria-label={`${output} minimum`}
        onChange={(event) => update({ min: parse(event.target.value) })}
      />
      <Input
        style={{ width: 82 }}
        inputMode="decimal"
        value={scale?.max ?? ''}
        placeholder="max"
        aria-label={`${output} maximum`}
        onChange={(event) => update({ max: parse(event.target.value) })}
      />
    </div>
  )
}

/** Drop scales for outputs that no longer exist, and incomplete ones. */
export function cleanScales(
  scales: Record<string, OutputScale> | undefined,
  outputs: string[],
): Record<string, OutputScale> | undefined {
  const out: Record<string, OutputScale> = {}
  for (const output of outputs) {
    const scale = scales?.[output]
    if (scale && scale.min !== undefined && scale.max !== undefined && scale.max > scale.min) out[output] = scale
  }
  return Object.keys(out).length ? out : undefined
}
