/** What the run actually cost, next to what it was projected to cost. */

import { memo, useMemo } from 'react'
import { Badge, EmptyState, Tooltip } from '@/components/ui/primitives'
import { isDecisionModelId, useDecisionModels } from '@/lib/decision'
import { modelShortName, tokens as formatTokens, titleCase, usd } from '@/lib/format'
import type { CostSummary, Estimate } from '@/lib/types'

/** Projection roles that are not agent names. */
const ROLE_LABELS: Record<string, string> = {
  question_book: 'Question book',
  tool: 'Tools',
}

const INPUT_ONLY_TIP = 'A decision model is billed for the state and questions it reads; its answers are free.'

const ROUTE_TIP: Record<string, string> = {
  direct: 'The projection assumed the record fits the Predictor input, so it has no Orchestrator or tool calls.',
  orchestrated: 'The projection assumed the record is orchestrated: a plan, its tool steps, and the Integrator.',
}

export interface CostPanelProps {
  cost: CostSummary
  estimate: Estimate
}

export const CostPanel = memo(function CostPanel({ cost, estimate }: CostPanelProps) {
  const actualLines = cost?.lines ?? []
  const projectedLines = estimate?.lines ?? []
  const decisionModels = useDecisionModels()
  const isDecision = (model: string) => isDecisionModelId(model, decisionModels.ids)

  const projectedByModel = useMemo(() => {
    const table = new Map<string, { tokens: number; usd: number | null }>()
    for (const line of projectedLines) {
      const current = table.get(line.model) ?? { tokens: 0, usd: line.usd === null ? null : 0 }
      const nextUsd = current.usd === null || line.usd === null ? null : current.usd + line.usd
      table.set(line.model, { tokens: current.tokens + line.total_tokens, usd: nextUsd })
    }
    return table
  }, [projectedLines])

  if (actualLines.length === 0 && projectedLines.length === 0) {
    return <EmptyState title="No cost recorded" body="Usage is reported per model as the agents call them." />
  }

  const spent = cost?.usd ?? null
  const projected = estimate?.usd ?? null
  const delta = spent !== null && projected !== null ? spent - projected : null

  return (
    <div className="stack gap-4">
      <div className="run-costtotals">
        <div className="stat">
          <span className="stat__label">Spent</span>
          <span className="stat__value">{usd(spent)}</span>
          <span className="stat__meta">{formatTokens(cost?.total_tokens ?? 0)} tokens</span>
        </div>
        <div className="stat">
          <span className="stat__label">Projected</span>
          <span className="stat__value">{usd(projected)}</span>
          <span className="stat__meta">{formatTokens(estimate?.total_tokens ?? 0)} tokens</span>
        </div>
        <div className="stat">
          <span className="stat__label">Difference</span>
          <span className="stat__value" style={{ color: delta !== null && delta > 0 ? 'var(--caution)' : 'var(--positive)' }}>
            {delta === null ? 'n/a' : `${delta > 0 ? '+' : ''}${usd(delta)}`}
          </span>
          <span className="stat__meta">
            {projected && spent !== null && projected > 0
              ? `${Math.round((spent / projected) * 100)}% of projection`
              : 'projection unpriced'}
          </span>
        </div>
      </div>

      {estimate?.route && (
        <div className="row gap-2 wrap">
          <span className="t-tiny muted">The projection assumed</span>
          <Tooltip content={ROUTE_TIP[estimate.route] ?? ''}>
            <Badge tone={estimate.route === 'direct' ? 'info' : 'neutral'}>
              {estimate.route === 'direct' ? 'Direct route' : 'Orchestrated'}
            </Badge>
          </Tooltip>
          {estimate.predictor_kind === 'decision' && (
            <Tooltip content={INPUT_ONLY_TIP}>
              <Badge tone="accent">Decision Predictor</Badge>
            </Tooltip>
          )}
        </div>
      )}

      <div className="grid grid--2">
        <div className="run-block">
          <span className="run-block__label">Actual, by model</span>
          {actualLines.length === 0 ? (
            <p className="t-tiny muted">No calls have been billed yet.</p>
          ) : (
            <div className="run-tablewrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="num">Prompt</th>
                    <th className="num">Completion</th>
                    <th className="num">USD</th>
                    <th className="num">Projected</th>
                  </tr>
                </thead>
                <tbody>
                  {actualLines.map((line) => {
                    const forecast = projectedByModel.get(line.model)
                    return (
                      <tr key={line.model}>
                        <td>
                          <span className="truncate" title={line.model}>
                            {modelShortName(line.model)}
                          </span>
                        </td>
                        <td className="num tabular">{formatTokens(line.prompt_tokens)}</td>
                        <td className="num tabular">
                          {isDecision(line.model) && line.completion_tokens === 0 ? (
                            <Tooltip content={INPUT_ONLY_TIP}>
                              <span className="muted">free</span>
                            </Tooltip>
                          ) : (
                            formatTokens(line.completion_tokens)
                          )}
                        </td>
                        <td className="num tabular">{usd(line.usd)}</td>
                        <td className="num tabular muted">{forecast ? usd(forecast.usd) : '-'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="run-block">
          <span className="run-block__label">Projected, by role</span>
          {projectedLines.length === 0 ? (
            <p className="t-tiny muted">This run was created without a projection.</p>
          ) : (
            <div className="run-tablewrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Role</th>
                    <th>Model</th>
                    <th className="num">Calls</th>
                    <th className="num">Tokens</th>
                    <th className="num">USD</th>
                  </tr>
                </thead>
                <tbody>
                  {projectedLines.map((line) => {
                    const inputOnly = isDecision(line.model) && line.completion_tokens === 0
                    return (
                      <tr key={`${line.role}-${line.model}`}>
                        <td>{ROLE_LABELS[line.role] ?? titleCase(line.role)}</td>
                        <td>
                          <span className="row gap-1" style={{ minWidth: 0 }}>
                            <span className="truncate" title={line.model}>
                              {modelShortName(line.model)}
                            </span>
                            {inputOnly && (
                              <Tooltip content={INPUT_ONLY_TIP}>
                                <Badge outline>input only</Badge>
                              </Tooltip>
                            )}
                          </span>
                        </td>
                        <td className="num tabular">{line.calls}</td>
                        <td className="num tabular">{formatTokens(line.total_tokens)}</td>
                        <td className="num tabular">
                          {line.price_known ? usd(line.usd) : <Badge tone="caution">unpriced</Badge>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
})
