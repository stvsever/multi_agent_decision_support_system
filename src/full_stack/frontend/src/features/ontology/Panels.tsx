/** The top "most deviating" strip and the domain coverage panel. */

import type { CSSProperties } from 'react'
import { Flame } from 'lucide-react'
import { Card, InfoDot } from '@/components/ui/primitives'
import { number, percent, signed, tokens } from '@/lib/format'
import type { DomainCoverage } from '@/lib/types'
import { MagnitudeBar } from './atoms'
import { bandColor, prettyLabel, type ExtremeChip } from './model'

export function ExtremesStrip({
  chips,
  clamp,
  selectedKey,
  selectedFeature,
  onSelect,
}: {
  chips: ExtremeChip[]
  clamp: number
  selectedKey: string | null
  selectedFeature: string | null
  onSelect: (key: string, feature: string | null) => void
}) {
  if (chips.length === 0) return null
  return (
    <section className="onto-extremes" aria-label="Most deviating measurements">
      <div className="onto-extremes__title">
        <Flame size={13} aria-hidden="true" />
        <span>Most deviating</span>
        <InfoDot label="About the ranking">
          Ranked by absolute deviation across every leaf and feature. The bar behind each chip is
          clamped to the 98th percentile of the participant's magnitudes so one stray value cannot
          flatten the rest; the printed number is always the true one.
        </InfoDot>
      </div>
      <div className="onto-extremes__rail">
        {chips.map((chip) => {
          const intensity = Math.min(1, Math.abs(chip.score) / clamp)
          const active = selectedKey === chip.nodeKey && selectedFeature === chip.feature
          return (
            <button
              key={chip.id}
              type="button"
              className="onto-extreme"
              data-active={active || undefined}
              onClick={() => onSelect(chip.nodeKey, chip.feature)}
              title={`${chip.label}: ${chip.score}`}
              style={
                {
                  '--tone': bandColor(chip.band),
                  '--intensity': `${8 + intensity * 84}%`,
                } as CSSProperties
              }
            >
              <span className="onto-extreme__fill" aria-hidden="true" />
              <span className="onto-extreme__label truncate">{chip.label}</span>
              <span className="onto-extreme__score tabular">{signed(chip.score, 2)}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

export function DomainCoveragePanel({
  coverage,
  onSelectDomain,
  activeDomain,
}: {
  coverage: Record<string, DomainCoverage>
  onSelectDomain: (domain: string) => void
  activeDomain: string
}) {
  const rows = Object.entries(coverage)
  if (rows.length === 0) return null
  const maxTokens = rows.reduce((max, [, value]) => Math.max(max, value.total_tokens ?? 0), 0)

  return (
    <Card
      title="Domain coverage"
      subtitle={`${rows.length} ${rows.length === 1 ? 'domain' : 'domains'}`}
      className="onto-rail__card"
    >
      <div className="stack gap-3">
        {rows.map(([id, row]) => {
          const fraction = row.total_leaves > 0 ? row.present_leaves / row.total_leaves : 0
          return (
            <button
              key={id}
              type="button"
              className="onto-coverage"
              data-active={activeDomain === id || undefined}
              onClick={() => onSelectDomain(activeDomain === id ? '' : id)}
            >
              <span className="onto-coverage__head">
                <span className="truncate semibold t-small">{prettyLabel(id)}</span>
                <span className="t-tiny muted tabular">{percent(fraction, 0)}</span>
              </span>
              <span className="onto-coverage__track" aria-hidden="true">
                <span
                  className="onto-coverage__fill"
                  style={{ width: `${Math.max(2, fraction * 100)}%` }}
                />
              </span>
              <span className="onto-coverage__meta t-tiny muted">
                <span className="tabular">
                  {number(row.present_leaves)} / {number(row.total_leaves)} leaves
                </span>
                <span className="tabular">{tokens(row.total_tokens)} tok</span>
              </span>
              {maxTokens > 0 && (
                <MagnitudeBar
                  value={row.total_tokens ?? 0}
                  max={maxTokens}
                  band={row.is_available ? 'normal' : 'missing'}
                  height={3}
                />
              )}
            </button>
          )
        })}
      </div>
    </Card>
  )
}
