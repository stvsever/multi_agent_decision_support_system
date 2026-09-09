/** Detail for the graph node the user picked in the flow canvas. */

import { X } from 'lucide-react'
import { memo, useMemo } from 'react'
import { Badge, Button, EmptyState } from '@/components/ui/primitives'
import { duration as formatDuration, tokens as formatTokens, titleCase } from '@/lib/format'
import type { GraphNode, RunStep } from '@/lib/types'
import { asNumber, asRecord, asStringList, asText, stepTone } from './runUtils'

export interface NodeInspectorProps {
  node: GraphNode | null
  steps: RunStep[]
  onClose: () => void
}

export const NodeInspector = memo(function NodeInspector({ node, steps, onClose }: NodeInspectorProps) {
  const step = useMemo(
    () => (node?.step_id === undefined ? undefined : steps.find((row) => row.id === node.step_id)),
    [node?.step_id, steps],
  )

  if (!node) {
    return (
      <div className="run-inspector run-inspector--empty">
        <EmptyState
          title="Nothing selected"
          body="Pick a node in the graph to see what that agent or tool was asked to do, and what it produced."
        />
      </div>
    )
  }

  const meta = asRecord(node.meta)
  const domains = asStringList(meta.input_domains)
  const expected = asText(meta.expected_output)
  const estimated = asNumber(meta.estimated_tokens)
  const actual = step?.tokens ?? asNumber(meta.actual_tokens)
  const executionMs = asNumber(meta.execution_time_ms)
  const seconds = step?.duration ?? (executionMs !== null ? executionMs / 1000 : null)
  const priority = asStringList(meta.priority_domains)
  const fusionStrategy = asText(meta.fusion_strategy)
  const totalSteps = asNumber(meta.total_steps)

  return (
    <div className="run-inspector">
      <header className="run-inspector__head">
        <div className="stack gap-1" style={{ minWidth: 0 }}>
          <div className="row gap-2 wrap">
            <span className="t-body semibold truncate">{node.label}</span>
            <Badge tone={node.type === 'agent' ? 'info' : 'neutral'}>{node.type === 'agent' ? 'Agent' : 'Tool'}</Badge>
            {node.role && <Badge tone="neutral">{titleCase(node.role)}</Badge>}
            {node.family && node.family !== 'other' && <Badge outline>{titleCase(node.family)}</Badge>}
            {step && <Badge tone={stepTone(step.status)}>{titleCase(step.status)}</Badge>}
          </div>
          <span className="t-tiny muted">
            Stage {node.stage + 1}
            {node.step_id !== undefined ? ` - step ${node.step_id}` : ''}
            {` - rank ${node.rank}`}
          </span>
        </div>
        <Button variant="ghost" size="sm" iconOnly icon={<X size={14} />} onClick={onClose} aria-label="Close detail" />
      </header>

      <div className="run-inspector__body">
        {node.detail && <p className="t-small secondary">{node.detail}</p>}

        {node.reasoning && (
          <div className="run-block">
            <span className="run-block__label">Why the orchestrator scheduled this</span>
            <p className="t-small secondary">{node.reasoning}</p>
          </div>
        )}

        {domains.length > 0 && (
          <div className="run-block">
            <span className="run-block__label">Input domains</span>
            <div className="row gap-1 wrap">
              {domains.map((domain) => (
                <Badge key={domain} outline mono>
                  {domain}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {expected && (
          <div className="run-block">
            <span className="run-block__label">Expected output</span>
            <p className="t-small secondary">{expected}</p>
          </div>
        )}

        {priority.length > 0 && (
          <div className="run-block">
            <span className="run-block__label">Priority domains</span>
            <div className="row gap-1 wrap">
              {priority.map((domain) => (
                <Badge key={domain} outline mono>
                  {domain}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {fusionStrategy && (
          <div className="run-block">
            <span className="run-block__label">Fusion strategy</span>
            <p className="t-small secondary">{fusionStrategy}</p>
          </div>
        )}

        <dl className="kv">
          {estimated !== null && (
            <>
              <dt>Estimated tokens</dt>
              <dd className="tabular">{formatTokens(estimated)}</dd>
            </>
          )}
          {actual !== null && actual !== undefined && (
            <>
              <dt>Actual tokens</dt>
              <dd className="tabular">
                {formatTokens(actual)}
                {estimated ? (
                  <span className="t-tiny muted"> {actual > estimated ? 'over' : 'under'} projection</span>
                ) : null}
              </dd>
            </>
          )}
          {seconds !== null && seconds !== undefined && (
            <>
              <dt>Execution time</dt>
              <dd className="tabular">{formatDuration(seconds)}</dd>
            </>
          )}
          {totalSteps !== null && (
            <>
              <dt>Planned steps</dt>
              <dd className="tabular">{totalSteps}</dd>
            </>
          )}
        </dl>

        {step?.error && (
          <div className="run-block">
            <span className="run-block__label" style={{ color: 'var(--critical)' }}>
              Error
            </span>
            <pre className="run-pre run-pre--critical">{step.error}</pre>
          </div>
        )}

        {step?.preview && (
          <div className="run-block">
            <span className="run-block__label">Output preview</span>
            <pre className="run-pre">{step.preview}</pre>
          </div>
        )}
      </div>
    </div>
  )
})
