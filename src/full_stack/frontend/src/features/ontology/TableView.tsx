/**
 * Flat, sortable list of every leaf and every feature.
 *
 * This is the "what is most abnormal" view, so it defaults to descending
 * magnitude and virtualises the same way the tree does.
 */

import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Table2 } from 'lucide-react'
import { EmptyState } from '@/components/ui/primitives'
import { compactNumber, signed } from '@/lib/format'
import { BandDot, DeviationBar, Highlight } from './atoms'
import {
  BAND_LABEL,
  chainOf,
  prettyLabel,
  type Filters,
  type OntologyIndex,
  type TableRow,
} from './model'
import { useVirtualRows } from './useOntologyIndex'

type SortKey = 'magnitude' | 'path' | 'label' | 'band'

export interface TableViewProps {
  index: OntologyIndex
  filters: Filters
  selected: string | null
  selectedFeature: string | null
  onSelect: (key: string, feature: string | null) => void
  rowHeight: number
}

export function TableView({
  index,
  filters,
  selected,
  selectedFeature,
  onSelect,
  rowHeight,
}: TableViewProps) {
  const [sort, setSort] = useState<SortKey>('magnitude')
  const [descending, setDescending] = useState(true)

  const rows = useMemo(() => {
    const needle = filters.search.trim().toLowerCase()
    const bands = filters.bands.length ? new Set(filters.bands) : null
    const filtered = index.rows.filter((row) => {
      if (filters.domain && row.domain !== filters.domain) return false
      if (bands && !bands.has(row.band)) return false
      if (filters.onlyMeasured && row.z === null) return false
      if (row.z === null ? filters.minAbs > 0 : Math.abs(row.z) < filters.minAbs) return false
      if (needle && !row.haystack.includes(needle)) return false
      return true
    })
    const direction = descending ? -1 : 1
    const compare = (a: TableRow, b: TableRow): number => {
      switch (sort) {
        case 'path':
          return a.pathText.localeCompare(b.pathText) * direction
        case 'label':
          return a.label.localeCompare(b.label) * direction
        case 'band':
          return (a.band < b.band ? -1 : a.band > b.band ? 1 : 0) * direction
        default: {
          const left = a.z === null ? -1 : Math.abs(a.z)
          const right = b.z === null ? -1 : Math.abs(b.z)
          return (left - right) * direction
        }
      }
    }
    return [...filtered].sort(compare)
  }, [index.rows, filters, sort, descending])

  const { ref, range, onScroll, totalHeight } = useVirtualRows(rows.length, rowHeight)

  const header = (key: SortKey, label: string, className?: string) => (
    <button
      type="button"
      className={`onto-table__head ${className ?? ''}`}
      aria-sort={sort === key ? (descending ? 'descending' : 'ascending') : 'none'}
      onClick={() => {
        if (sort === key) setDescending((value) => !value)
        else {
          setSort(key)
          setDescending(key === 'magnitude')
        }
      }}
    >
      {label}
      {sort === key &&
        (descending ? (
          <ArrowDown size={11} aria-hidden="true" />
        ) : (
          <ArrowUp size={11} aria-hidden="true" />
        ))}
    </button>
  )

  const slice = rows.slice(range.start, range.end)

  return (
    <div className="onto-table">
      <div className="onto-table__bar">
        <span className="t-tiny muted">
          {compactNumber(rows.length)} of {compactNumber(index.rows.length)} measurements
        </span>
      </div>
      <div className="onto-table__row onto-table__row--head" role="presentation">
        {header('path', 'Path')}
        {header('label', 'Measurement')}
        <span className="onto-table__head onto-table__head--static">Value</span>
        {header('magnitude', 'z', 'onto-table__head--num')}
        <span className="onto-table__head onto-table__head--static">Deviation</span>
        {header('band', 'Band')}
      </div>

      {rows.length === 0 ? (
        <EmptyState
          icon={<Table2 size={20} />}
          title="No measurement matches"
          body="Clear a band chip or lower the deviation threshold."
        />
      ) : (
        <div ref={ref} className="onto-table__scroll" onScroll={onScroll}>
          <div style={{ height: totalHeight, position: 'relative' }} role="presentation">
            <div
              role="presentation"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                transform: `translateY(${range.start * rowHeight}px)`,
              }}
            >
              {slice.map((row) => (
                <Row
                  key={row.id}
                  row={row}
                  index={index}
                  query={filters.search}
                  height={rowHeight}
                  selected={selected === row.nodeKey && selectedFeature === row.feature}
                  onSelect={() => onSelect(row.nodeKey, row.feature)}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Row({
  row,
  index,
  query,
  height,
  selected,
  onSelect,
}: {
  row: TableRow
  index: OntologyIndex
  query: string
  height: number
  selected: boolean
  onSelect: () => void
}) {
  const chain = chainOf(index, row.nodeKey)
  const trail = row.feature ? chain : chain.slice(0, -1)
  const shown = trail.slice(-3)

  return (
    <button
      type="button"
      className="onto-table__row"
      data-selected={selected || undefined}
      style={{ height }}
      onClick={onSelect}
      title={row.pathText}
    >
      <span className="onto-table__path">
        {trail.length > shown.length && <span className="onto-pill onto-pill--more">...</span>}
        {shown.map((step) => (
          <span className="onto-pill" key={step.key}>
            {prettyLabel(step.node.label)}
          </span>
        ))}
      </span>
      <span className="onto-table__label truncate">
        <Highlight text={row.label} query={query} />
        {row.feature && <span className="onto-table__tag">feature</span>}
      </span>
      <span className="onto-table__value truncate" title={row.value}>
        {row.value}
      </span>
      <span className="onto-table__num tabular">{row.z === null ? '-' : signed(row.z)}</span>
      <span className="onto-table__dev">
        <DeviationBar score={row.z} clamp={index.clamp} width={72} />
      </span>
      <span className="onto-table__band">
        <BandDot band={row.band} size={7} />
        <span className="truncate">{BAND_LABEL[row.band]}</span>
      </span>
    </button>
  )
}
