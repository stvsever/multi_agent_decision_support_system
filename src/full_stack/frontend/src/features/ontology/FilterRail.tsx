/** Left rail: search, band chips, threshold, domain and coverage filters. */

import type { CSSProperties } from 'react'
import { Search, X } from 'lucide-react'
import { Button, Card, Field, Input, Select, Toggle } from '@/components/ui/primitives'
import { compactNumber } from '@/lib/format'
import type { Band } from '@/lib/types'
import {
  BAND_LABEL,
  BAND_ORDER,
  bandColor,
  emptyFilters,
  filtersActive,
  type Filters,
  type OntologyIndex,
} from './model'

export interface FilterRailProps {
  index: OntologyIndex
  filters: Filters
  onChange: (next: Filters) => void
  matchCount: number | null
}

export function FilterRail({ index, filters, onChange, matchCount }: FilterRailProps) {
  const patch = (part: Partial<Filters>) => onChange({ ...filters, ...part })

  const toggleBand = (band: Band) => {
    const next = filters.bands.includes(band)
      ? filters.bands.filter((value) => value !== band)
      : [...filters.bands, band]
    patch({ bands: next })
  }

  const dirty = filtersActive(filters) || filters.search !== ''

  return (
    <Card
      title="Filter"
      subtitle={
        matchCount === null
          ? `${compactNumber(index.atomCount)} measurements`
          : `${compactNumber(matchCount)} nodes match`
      }
      actions={
        dirty ? (
          <Button size="sm" variant="ghost" onClick={() => onChange(emptyFilters())}>
            Clear
          </Button>
        ) : undefined
      }
      className="onto-rail__card"
    >
      <div className="stack gap-4">
        <Field label="Search">
          <div className="onto-search">
            <Search size={14} className="onto-search__icon" aria-hidden="true" />
            <Input
              value={filters.search}
              placeholder="Label, id or feature"
              aria-label="Search the ontology"
              onChange={(event) => patch({ search: event.target.value })}
            />
            {filters.search && (
              <button
                type="button"
                className="onto-search__clear"
                aria-label="Clear search"
                onClick={() => patch({ search: '' })}
              >
                <X size={13} />
              </button>
            )}
          </div>
        </Field>

        <Field
          label="Bands"
          hint={filters.bands.length ? `${filters.bands.length} selected` : 'All bands'}
        >
          <div className="onto-bands">
            {BAND_ORDER.map((band) => {
              const count = index.bandCounts[band]
              const on = filters.bands.includes(band)
              return (
                <button
                  key={band}
                  type="button"
                  className="onto-band"
                  aria-pressed={on}
                  disabled={count === 0}
                  onClick={() => toggleBand(band)}
                  style={{ '--tone': bandColor(band) } as CSSProperties}
                >
                  <span className="onto-band__swatch" aria-hidden="true" />
                  <span className="onto-band__label truncate">{BAND_LABEL[band]}</span>
                  <span className="onto-band__count tabular">{compactNumber(count)}</span>
                </button>
              )
            })}
          </div>
        </Field>

        <Field
          label="Deviation threshold"
          hint={
            filters.minAbs === 0
              ? 'Showing every node'
              : `Showing nodes deviating at least ${filters.minAbs.toFixed(2)}`
          }
        >
          <div className="row gap-3">
            <input
              type="range"
              className="slider grow"
              min={0}
              max={5}
              step={0.25}
              value={filters.minAbs}
              aria-label="Minimum absolute z"
              onChange={(event) => patch({ minAbs: Number(event.target.value) })}
            />
            <span className="t-tiny tabular muted" style={{ width: 34, textAlign: 'right' }}>
              {filters.minAbs.toFixed(2)}
            </span>
          </div>
        </Field>

        <Field label="Domain">
          <Select
            value={filters.domain}
            aria-label="Domain filter"
            onChange={(event) => patch({ domain: event.target.value })}
          >
            <option value="">All domains</option>
            {index.domains.map((domain) => (
              <option key={domain.id} value={domain.id}>
                {domain.label}
              </option>
            ))}
          </Select>
        </Field>

        <div className="row between gap-3">
          <div className="stack" style={{ gap: 1 }}>
            <span className="t-small semibold">Only measured</span>
            <span className="t-tiny muted">Hide nodes with no present leaves</span>
          </div>
          <Toggle
            checked={filters.onlyMeasured}
            label="Only measured"
            onChange={(value) => patch({ onlyMeasured: value })}
          />
        </div>
      </div>
    </Card>
  )
}
