/**
 * One leaf, across the selected participants.
 *
 * The service infers what kind of variable it is looking at, and the shape it
 * reports decides the chart: a smoothed outline over counts of a whole number
 * would be a lie, and a nominal category has no order to preserve, so neither
 * borrows the other's treatment. Small cohorts are labelled as small rather
 * than dressed up as a reference distribution.
 */

import { useMemo } from 'react'
import { curveBasis, line } from 'd3-shape'
import { BarChart3, Ban, Info } from 'lucide-react'
import { ApiError } from '@/lib/api'
import { Badge, Callout, Card, Disclosure, EmptyState, Skeleton } from '@/components/ui/primitives'
import { number, signed } from '@/lib/format'
import { useDistribution } from '@/lib/hooks'
import type { CategoryCount, DistributionResponse, NumericSummary, VariableType } from '@/lib/types'
import { formatValue } from './model'
import { useMeasure } from './useOntologyIndex'

const TYPE_LABEL: Record<VariableType, string> = {
  continuous: 'Continuous',
  integer: 'Whole numbers',
  ordinal: 'Ordered categories',
  nominal: 'Categories',
  missing: 'Not measured',
}

/** Below this a cohort describes itself and nothing else. */
const THIN_COHORT = 12
const CHART_H = 108

export interface DistributionPanelProps {
  directories: string[]
  path: string[] | null
  label: string
  /** Participant whose own value is marked inside the cohort, if any. */
  focusId: string | null
}

export function DistributionPanel({ directories, path, label, focusId }: DistributionPanelProps) {
  const cohort = directories.length > 1
  const query = useDistribution(cohort ? directories : [], path)

  if (!path) return null

  if (!cohort) {
    return (
      <Card title="Distribution" subtitle={label} className="onto-rail__card">
        <Callout tone="neutral" icon={<Info size={15} />}>
          A distribution needs a cohort. Add participants at the top of the page and this leaf will
          show where this one sits among them.
        </Callout>
      </Card>
    )
  }

  if (query.isLoading) {
    return (
      <Card title="Distribution" subtitle={label} className="onto-rail__card">
        <div className="stack gap-3">
          <Skeleton height={12} width={120} />
          <Skeleton height={CHART_H} />
        </div>
      </Card>
    )
  }

  if (query.isError) {
    const error = query.error
    const missing = error instanceof ApiError && error.status === 404
    return (
      <Card title="Distribution" subtitle={label} className="onto-rail__card">
        <Callout
          tone={missing ? 'neutral' : 'caution'}
          icon={<Ban size={15} />}
          title={missing ? 'Distributions are not available' : 'The distribution failed to load'}
        >
          {missing
            ? 'This dashboard service does not serve cohort distributions. Everything else on this page still works.'
            : ((error as Error)?.message ?? 'The dashboard service returned an error.')}
        </Callout>
      </Card>
    )
  }

  const data = query.data
  if (!data) return null

  return <Body data={data} label={label} focusId={focusId} />
}

function Body({
  data,
  label,
  focusId,
}: {
  data: DistributionResponse
  label: string
  focusId: string | null
}) {
  const rows = data.rows ?? []
  const measured = rows.filter((row) => row.value !== null && row.value !== undefined)
  const focusRow = focusId ? rows.find((row) => row.participant_id === focusId) : undefined
  const focusValue = typeof focusRow?.value === 'number' ? focusRow.value : null

  return (
    <Card
      title="Distribution"
      subtitle={label}
      actions={<Badge tone="neutral" outline>{TYPE_LABEL[data.variable_type]}</Badge>}
      className="onto-rail__card"
    >
      <div className="stack gap-4">
        <span className="t-tiny muted">
          {number(measured.length)} of {number(data.participant_count)} participants measured
        </span>

        {data.variable_type === 'missing' || measured.length === 0 ? (
          <EmptyState
            icon={<Ban size={18} />}
            title="Not measured in this cohort"
            body="No participant in the selection carries a value for this leaf."
          />
        ) : measured.length === 1 ? (
          <Callout tone="neutral" icon={<Info size={15} />} title="One value only">
            {measured[0].participant_id} reports {formatValue(measured[0].value)}. A single
            observation is not a distribution, so nothing is drawn.
          </Callout>
        ) : data.variable_type === 'continuous' && data.numeric ? (
          <Continuous numeric={data.numeric} mark={focusValue} />
        ) : data.variable_type === 'integer' ? (
          <Discrete rows={measured.map((row) => row.value)} numeric={data.numeric} mark={focusValue} />
        ) : (
          <Categories
            categories={data.categories ?? []}
            sort={data.variable_type === 'nominal'}
            mark={focusRow ? formatValue(focusRow.value) : null}
          />
        )}

        {focusRow && measured.length > 1 && (
          <div className="onto-mark-note">
            <span className="onto-mark-note__dot" aria-hidden="true" />
            <span className="t-tiny">
              <b>{focusRow.participant_id}</b>: {formatValue(focusRow.value)}
              {focusRow.z_score === null ? '' : `, z ${signed(focusRow.z_score)}`}
            </span>
          </div>
        )}

        {measured.length > 1 && measured.length < THIN_COHORT && (
          <span className="t-micro faint">
            {measured.length} participants describe these {measured.length} participants. Read the
            spread as a comparison inside the selection, not as a reference range.
          </span>
        )}

        {measured.length > 1 && (
          <Disclosure title="Values" subtitle={`${number(rows.length)} participants`}>
            <div className="onto-scrollbox">
              <table className="table onto-features">
                <thead>
                  <tr>
                    <th>Participant</th>
                    <th>Value</th>
                    <th className="num">z</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.participant_id} data-selected={row.participant_id === focusId || undefined}>
                      <td className="truncate">{row.participant_id}</td>
                      <td className="truncate">{formatValue(row.value)}</td>
                      <td className="num tabular">
                        {row.z_score === null ? '-' : signed(row.z_score)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Disclosure>
        )}
      </div>
    </Card>
  )
}

/* --- Continuous ------------------------------------------------------------ */

function Continuous({ numeric, mark }: { numeric: NumericSummary; mark: number | null }) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const width = Math.max(180, size.width || 260)
  const bins = numeric.bins ?? []
  const low = bins.length ? bins[0].start : (numeric.min ?? 0)
  const high = bins.length ? bins[bins.length - 1].end : (numeric.max ?? 1)
  const span = high - low || 1
  const max = bins.reduce((top, bin) => Math.max(top, bin.count), 0)
  const plot = CHART_H - 26
  const at = (value: number) => ((value - low) / span) * (width - 2) + 1

  // The outline runs through the top of every bar, so it describes the same
  // counts rather than a second estimate of them.
  const outline = useMemo(() => {
    if (bins.length < 3 || max <= 0) return null
    const points: [number, number][] = bins.map((bin) => [
      ((bin.start + bin.end) / 2 - low) / span,
      bin.count / max,
    ])
    const path = line<[number, number]>()
      .x((point) => point[0] * (width - 2) + 1)
      .y((point) => plot - point[1] * (plot - 6))
      .curve(curveBasis)
    return path([[0, 0], ...points, [1, 0]])
  }, [bins, low, span, max, width, plot])

  return (
    <div ref={ref} className="onto-dist">
      <svg width={width} height={CHART_H} role="img" aria-label="Distribution across the cohort">
        {bins.map((bin) => {
          const height = max > 0 && bin.count > 0 ? Math.max(2, (bin.count / max) * (plot - 6)) : 0
          const x = at(bin.start)
          const right = at(bin.end)
          return (
            <rect
              key={bin.start}
              className="onto-dist__bar"
              x={x + 0.5}
              y={plot - height}
              width={Math.max(1, right - x - 1)}
              height={height}
              rx={1.5}
            >
              <title>{`${bin.start.toFixed(2)} to ${bin.end.toFixed(2)}: ${bin.count}`}</title>
            </rect>
          )
        })}
        {outline && <path className="onto-dist__outline" d={outline} />}
        <line className="onto-dist__axis" x1={0} x2={width} y1={plot} y2={plot} />

        <QuantileStrip numeric={numeric} at={at} y={plot + 8} />

        {mark !== null && (
          <g className="onto-dist__mark">
            <line x1={at(mark)} x2={at(mark)} y1={2} y2={plot + 14} />
            <circle cx={at(mark)} cy={2} r={3} />
          </g>
        )}

        <text className="onto-dist__tick" x={1} y={CHART_H - 2}>
          {low.toFixed(2)}
        </text>
        <text className="onto-dist__tick" x={width - 1} y={CHART_H - 2} textAnchor="end">
          {high.toFixed(2)}
        </text>
      </svg>
    </div>
  )
}

function QuantileStrip({
  numeric,
  at,
  y,
}: {
  numeric: NumericSummary
  at: (value: number) => number
  y: number
}) {
  const { min, q1, median, q3, max } = numeric
  if (min === null || q1 === null || median === null || q3 === null || max === null) return null
  return (
    <g className="onto-dist__box">
      <line x1={at(min)} x2={at(max)} y1={y + 4} y2={y + 4} />
      <rect x={at(q1)} y={y} width={Math.max(1.5, at(q3) - at(q1))} height={9} rx={2} />
      <line className="onto-dist__median" x1={at(median)} x2={at(median)} y1={y - 1} y2={y + 10} />
      <title>{`min ${min.toFixed(2)}, q1 ${q1.toFixed(2)}, median ${median.toFixed(2)}, q3 ${q3.toFixed(2)}, max ${max.toFixed(2)}`}</title>
    </g>
  )
}

/* --- Whole numbers --------------------------------------------------------- */

/**
 * Counts of a whole number get one bar per value. Falling back to the binned
 * summary only happens when the range is too wide to enumerate, and even then
 * the bars stay separate rather than being smoothed.
 */
function Discrete({
  rows,
  numeric,
  mark,
}: {
  rows: (number | string | boolean | null)[]
  numeric?: NumericSummary
  mark: number | null
}) {
  const counts = useMemo(() => {
    const tally = new Map<number, number>()
    for (const value of rows) {
      if (typeof value !== 'number' || !Number.isFinite(value)) continue
      tally.set(value, (tally.get(value) ?? 0) + 1)
    }
    return [...tally.entries()].sort((a, b) => a[0] - b[0])
  }, [rows])

  const [ref, size] = useMeasure<HTMLDivElement>()
  const width = Math.max(180, size.width || 260)
  const plot = CHART_H - 20

  if (counts.length === 0) return null

  const wide = counts.length > 30
  const bars = wide
    ? (numeric?.bins ?? []).map((bin) => ({
        key: bin.start,
        label: `${bin.start.toFixed(0)} to ${bin.end.toFixed(0)}`,
        count: bin.count,
        hit: mark !== null && mark >= bin.start && mark < bin.end,
      }))
    : counts.map(([value, count]) => ({
        key: value,
        label: String(value),
        count,
        hit: mark !== null && mark === value,
      }))

  const max = bars.reduce((top, bar) => Math.max(top, bar.count), 0)
  const step = width / Math.max(1, bars.length)
  const barWidth = Math.max(2, Math.min(28, step - 4))

  return (
    <div ref={ref} className="onto-dist">
      <svg width={width} height={CHART_H} role="img" aria-label="Counts across the cohort">
        {bars.map((bar, at) => {
          const height = max > 0 && bar.count > 0 ? Math.max(2, (bar.count / max) * (plot - 8)) : 0
          const x = at * step + (step - barWidth) / 2
          return (
            <g key={bar.key}>
              <rect
                className="onto-dist__bar"
                data-hit={bar.hit || undefined}
                x={x}
                y={plot - height}
                width={barWidth}
                height={height}
                rx={2}
              >
                <title>{`${bar.label}: ${bar.count}`}</title>
              </rect>
              {bars.length <= 14 && (
                <text className="onto-dist__tick" x={x + barWidth / 2} y={CHART_H - 4} textAnchor="middle">
                  {bar.label}
                </text>
              )}
            </g>
          )
        })}
        <line className="onto-dist__axis" x1={0} x2={width} y1={plot} y2={plot} />
      </svg>
    </div>
  )
}

/* --- Categories ------------------------------------------------------------ */

function Categories({
  categories,
  sort,
  mark,
}: {
  categories: CategoryCount[]
  /** Nominal has no natural order, so it is ranked. Ordinal keeps its own. */
  sort: boolean
  mark: string | null
}) {
  const rows = useMemo(
    () => (sort ? [...categories].sort((a, b) => b.count - a.count) : categories),
    [categories, sort],
  )
  const max = rows.reduce((top, row) => Math.max(top, row.count), 0)
  if (rows.length === 0) return null

  return (
    <div className="stack gap-2">
      {rows.map((row) => (
        <div
          key={row.label}
          className="onto-cat"
          data-hit={mark !== null && mark === row.label ? '' : undefined}
        >
          <span className="onto-cat__label truncate" title={row.label}>
            {row.label}
          </span>
          <span className="onto-cat__track" aria-hidden="true">
            <span
              className="onto-cat__fill"
              style={{ width: `${max > 0 ? Math.max(2, (row.count / max) * 100) : 0}%` }}
            />
          </span>
          <span className="onto-cat__count t-micro tabular muted">{number(row.count)}</span>
        </div>
      ))}
    </div>
  )
}

export const DISTRIBUTION_ICON = BarChart3
