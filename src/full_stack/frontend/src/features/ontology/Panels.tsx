/** Left rail: how much of each domain the participant actually carries. */

import { Card } from '@/components/ui/primitives'
import { number, percent, tokens } from '@/lib/format'
import type { DomainCoverage } from '@/lib/types'
import { MagnitudeBar } from './atoms'
import { prettyLabel } from './model'

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
