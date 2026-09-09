/** Every step in execution order, grouped by iteration and by parallel batch. */

import clsx from 'clsx'
import { AlertTriangle, Check, ChevronRight, Loader2, Wrench } from 'lucide-react'
import { memo, useMemo, useState } from 'react'
import { Badge, EmptyState } from '@/components/ui/primitives'
import { duration as formatDuration, tokens as formatTokens } from '@/lib/format'
import type { RunGraph, RunStep } from '@/lib/types'
import { groupByIteration, ranksFromGraph, stepTone } from './runUtils'

export interface TimelinePanelProps {
  steps: RunStep[]
  history: RunStep[]
  graph: RunGraph | null
}

export const TimelinePanel = memo(function TimelinePanel({ steps, history, graph }: TimelinePanelProps) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({})
  const ranks = useMemo(() => ranksFromGraph(graph), [graph])
  const iterations = useMemo(
    () => groupByIteration([...history, ...steps], ranks),
    [history, steps, ranks],
  )

  if (iterations.length === 0) {
    return (
      <EmptyState
        title="No steps yet"
        body="The orchestrator has not dispatched anything. Steps appear here the moment they start."
      />
    )
  }

  const toggle = (id: number) => setExpanded((prev) => ({ ...prev, [id]: !prev[id] }))

  return (
    <div className="run-timeline">
      {iterations.map((iteration) => (
        <section key={iteration.iteration} className="run-timeline__iteration">
          <header className="run-timeline__head">
            <span className="eyebrow">Iteration {iteration.iteration}</span>
            <span className="t-tiny muted tabular">
              {iteration.steps.length} steps -{' '}
              {formatTokens(iteration.steps.reduce((sum, step) => sum + (step.tokens ?? 0), 0))} tokens
            </span>
          </header>

          {iteration.groups.map((group) => {
            const parallel = group.steps.length > 1
            return (
              <div key={`${iteration.iteration}-${group.key}`} className={clsx('run-batch', parallel && 'run-batch--parallel')}>
                {parallel && (
                  <span className="run-batch__label">
                    {group.steps.length} in parallel
                    {group.rank !== null ? ` - rank ${group.rank}` : ''}
                  </span>
                )}
                <ul className="run-batch__steps">
                  {group.steps.map((step) => (
                    <StepRow
                      key={`${iteration.iteration}-${step.id}`}
                      step={step}
                      open={Boolean(expanded[step.id])}
                      onToggle={() => toggle(step.id)}
                    />
                  ))}
                </ul>
              </div>
            )
          })}
        </section>
      ))}
    </div>
  )
})

function StepIcon({ status }: { status: RunStep['status'] }) {
  if (status === 'complete') return <Check size={13} />
  if (status === 'failed') return <AlertTriangle size={13} />
  if (status === 'repairing') return <Wrench size={13} />
  return <Loader2 size={13} className="spin" />
}

function StepRow({ step, open, onToggle }: { step: RunStep; open: boolean; onToggle: () => void }) {
  const detail = step.preview || step.error || step.desc
  const expandable = Boolean(detail)
  return (
    <li className={clsx('run-step', `run-step--${step.status}`)}>
      <button
        type="button"
        className="run-step__trigger"
        onClick={onToggle}
        aria-expanded={expandable ? open : undefined}
        disabled={!expandable}
      >
        <span className="run-step__icon" aria-hidden>
          <StepIcon status={step.status} />
        </span>
        <span className="run-step__text">
          <span className="row gap-2" style={{ minWidth: 0 }}>
            <span className="t-small semibold truncate">{step.tool}</span>
            <span className="t-micro faint tabular">#{step.id}</span>
          </span>
          <span className="t-tiny muted truncate">{step.desc}</span>
        </span>
        <span className="run-step__meta">
          <Badge tone={stepTone(step.status)}>{step.status}</Badge>
          <span className="t-tiny muted tabular">{formatTokens(step.tokens ?? 0)}</span>
          <span className="t-tiny muted tabular">{step.duration ? formatDuration(step.duration) : '-'}</span>
        </span>
        {expandable && <ChevronRight size={14} className={clsx('run-step__chevron', open && 'run-step__chevron--open')} />}
      </button>
      {open && expandable && (
        <div className="run-step__panel">
          {step.desc && <p className="t-tiny secondary">{step.desc}</p>}
          {step.error && <pre className="run-pre run-pre--critical">{step.error}</pre>}
          {step.preview && <pre className="run-pre">{step.preview}</pre>}
        </div>
      )}
    </li>
  )
}
