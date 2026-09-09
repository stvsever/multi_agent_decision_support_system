/** The critic's evaluation: score, checklist, and what it wants improved. */

import clsx from 'clsx'
import { AlertTriangle, Check, X } from 'lucide-react'
import { memo } from 'react'
import { Badge, Callout, EmptyState } from '@/components/ui/primitives'
import { percent, titleCase } from '@/lib/format'
import { asArray, asNumber, asRecord, asStringList, asText, verdictTone, type Tone } from './runUtils'

export interface CriticPanelProps {
  critic: Record<string, unknown> | null | undefined
}

const PRIORITY_TONE: Record<string, Tone> = { HIGH: 'critical', MEDIUM: 'caution', LOW: 'neutral' }

export const CriticPanel = memo(function CriticPanel({ critic }: CriticPanelProps) {
  if (!critic || Object.keys(critic).length === 0) {
    return (
      <EmptyState
        title="No verdict yet"
        body="The critic scores the prediction once it exists. If it fails, the orchestrator replans and the run iterates."
      />
    )
  }

  const verdict = asText(critic.verdict) || 'UNKNOWN'
  const confidence = asNumber(critic.confidence)
  const passed = asNumber(critic.passed)
  const total = asNumber(critic.total)
  const summary = asText(critic.summary)
  const composite = asNumber(critic.composite_score)
  const breakdown = asRecord(critic.score_breakdown)
  const checklist = asRecord(critic.checklist)
  const active = asStringList(checklist.active_checks)
  const weaknesses = asStringList(critic.weaknesses)
  const missed = asStringList(critic.domains_missed)
  const suggestions = asArray(critic.improvement_suggestions).map((row) => asRecord(row))
  const fallbackUsed = critic.fallback_used === true
  const iteration = asNumber(critic.iteration)

  const checkKeys = (active.length > 0 ? active : Object.keys(checklist)).filter(
    (key) => key !== 'active_checks' && typeof checklist[key] === 'boolean',
  )

  return (
    <div className="stack gap-4">
      {fallbackUsed && (
        <Callout
          tone="critical"
          icon={<AlertTriangle size={15} />}
          title="A deterministic fallback produced this prediction"
        >
          <div className="stack gap-1">
            <span>{asText(critic.fallback_reason) || 'The predictor could not produce a model-backed result.'}</span>
            {asText(critic.fallback_recommendation) && (
              <span className="t-tiny">{asText(critic.fallback_recommendation)}</span>
            )}
          </div>
        </Callout>
      )}

      <div className="run-verdict">
        <div className="stack gap-1" style={{ minWidth: 0 }}>
          <span className="stat__label">Verdict</span>
          <div className="row gap-2 wrap">
            <Badge tone={verdictTone(verdict)}>{verdict}</Badge>
            {passed !== null && total !== null && (
              <span className="t-tiny muted tabular">
                {passed} of {total} checks
              </span>
            )}
            {confidence !== null && <span className="t-tiny muted tabular">confidence {confidence.toFixed(2)}</span>}
            {iteration !== null && <Badge outline>Iteration {iteration}</Badge>}
          </div>
        </div>
        {composite !== null && (
          <div className="run-prob">
            <span className="run-prob__value tabular">{composite.toFixed(2)}</span>
            <span className="run-meter" aria-hidden>
              <span
                className={clsx('run-meter__fill', composite < 0.7 && 'run-meter__fill--warn')}
                style={{ width: `${Math.round(Math.max(0, Math.min(1, composite)) * 100)}%` }}
              />
            </span>
            <span className="t-micro muted">composite score</span>
          </div>
        )}
      </div>

      {summary && <p className="t-small secondary">{summary}</p>}

      {Object.keys(breakdown).length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Score breakdown</span>
          <div className="stack gap-1">
            {Object.entries(breakdown).map(([key, raw]) => {
              const value = asNumber(raw) ?? 0
              return (
                <div key={key} className="run-probrow">
                  <span className="t-tiny truncate">{titleCase(key)}</span>
                  <span className="run-meter" aria-hidden>
                    <span
                      className={clsx('run-meter__fill', value < 0.7 && 'run-meter__fill--warn')}
                      style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }}
                    />
                  </span>
                  <span className="t-tiny muted tabular">{percent(value, 0)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {checkKeys.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Checklist</span>
          <ul className="run-checks">
            {checkKeys.map((key) => {
              const ok = checklist[key] === true
              return (
                <li key={key} className={clsx('run-check', ok ? 'run-check--pass' : 'run-check--fail')}>
                  <span className="run-check__icon" aria-hidden>
                    {ok ? <Check size={12} /> : <X size={12} />}
                  </span>
                  <span className="t-tiny truncate">{titleCase(key)}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {weaknesses.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Weaknesses</span>
          <ul className="run-bullets">
            {weaknesses.map((item, index) => (
              <li key={index} className="t-small secondary">
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Improvements requested</span>
          <ul className="stack gap-2">
            {suggestions.map((row, index) => {
              const priority = asText(row.priority).toUpperCase() || 'MEDIUM'
              return (
                <li key={index} className="run-suggestion">
                  <Badge tone={PRIORITY_TONE[priority] ?? 'neutral'}>{priority}</Badge>
                  <div className="stack gap-1" style={{ minWidth: 0 }}>
                    <span className="t-small semibold">{asText(row.issue)}</span>
                    <span className="t-tiny secondary">{asText(row.suggestion)}</span>
                  </div>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {missed.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Domains missed</span>
          <div className="row gap-1 wrap">
            {missed.map((domain) => (
              <Badge key={domain} tone="caution" mono>
                {domain}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  )
})
