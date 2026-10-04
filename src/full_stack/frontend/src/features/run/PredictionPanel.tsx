/**
 * The predictor's output.
 *
 * The payload shape follows the task: a classification node carries labelled
 * probabilities, a regression node carries values, and a hierarchical task
 * nests child nodes under the root. Everything is read defensively so a novel
 * payload still renders.
 */

import clsx from 'clsx'
import { memo, useMemo, type ReactNode } from 'react'
import { Badge, Disclosure, EmptyState, Tooltip } from '@/components/ui/primitives'
import { DECISION_TOOLTIP } from '@/lib/decision'
import { percent, titleCase, tokens as formatTokens, usd } from '@/lib/format'
import { asArray, asNumber, asRecord, asStringList, asText } from './runUtils'

/** Above this mean change between presentation orders an answer reads as unstable. */
/** Instability is reported as a multiple of its threshold; above 1 the Critic rejects. */
const INSTABILITY_CAUTION = 1

/** What each node needs from the run as a whole while it renders. */
interface NodeContext {
  /** Unit per regression output, keyed by node id. */
  units: Map<string, Record<string, string>>
  /** Per-node decision statistics from the decision report. */
  perNode: Record<string, unknown>
}

export interface PredictionPanelProps {
  prediction: Record<string, unknown> | null | undefined
}

export const PredictionPanel = memo(function PredictionPanel({ prediction }: PredictionPanelProps) {
  const context = useMemo<NodeContext>(() => {
    const payload = asRecord(prediction?.payload)
    return {
      units: unitsByNode(asRecord(payload.prediction_task_spec)),
      perNode: asRecord(asRecord(asRecord(payload.decision_report).quality).per_node),
    }
  }, [prediction])

  if (!prediction || Object.keys(prediction).length === 0) {
    return (
      <EmptyState
        title="No prediction yet"
        body="The predictor runs after integration. Its output lands here the moment it is produced."
      />
    )
  }

  const label = asText(prediction.result) || asText(prediction.label) || 'Prediction ready'
  const probability = asNumber(prediction.prob)
  const confidence = asText(prediction.confidence)
  const payload = asRecord(prediction.payload)
  const root = asRecord(payload.root_prediction)
  const flat = asArray(payload.flat_predictions).map((row) => asRecord(row))
  const summary = asText(payload.clinical_summary) || asText(payload.summary)
  const findings = asArray(payload.key_findings)
  const kind = asText(payload.primary_output_kind)
  const domains = asStringList(payload.domains_processed)
  const reasoning = asStringList(payload.reasoning_chain)
  const uncertainty = asStringList(payload.uncertainty_factors)
  const usedTokens = asNumber(payload.total_tokens_used)
  const predictorKind = asText(payload.predictor_kind)
  const predictorModel = asText(payload.predictor_model)
  const inputRoute = asText(payload.input_route)
  const report = asRecord(payload.decision_report)
  const outputKind = kind || deriveOutputKind(root, flat)

  return (
    <div className="stack gap-4">
      <div className="run-verdict">
        <div className="stack gap-1" style={{ minWidth: 0 }}>
          <span className="stat__label">Predicted</span>
          <span className="t-h2 truncate">{label}</span>
          <div className="row gap-2 wrap">
            {outputKind && outputKind !== 'unknown' && <Badge tone="neutral">{titleCase(outputKind)}</Badge>}
            {confidence && <Badge tone="info">{titleCase(confidence)} confidence</Badge>}
            {predictorKind === 'decision' && (
              <Tooltip content={predictorModel ? `${DECISION_TOOLTIP} Model: ${predictorModel}.` : DECISION_TOOLTIP}>
                <Badge tone="accent">Decision model</Badge>
              </Tooltip>
            )}
            {(inputRoute === 'direct' || inputRoute === 'orchestrated') && (
              <Badge outline>{inputRoute === 'direct' ? 'Direct route' : 'Orchestrated'}</Badge>
            )}
            {usedTokens !== null && <Badge outline>{formatTokens(usedTokens)} tokens</Badge>}
          </div>
        </div>
        {probability !== null && (
          <div className="run-prob">
            <span className="run-prob__value tabular">{percent(probability, 1)}</span>
            <span className="run-meter" aria-hidden>
              <span className="run-meter__fill" style={{ width: `${Math.round(Math.max(0, Math.min(1, probability)) * 100)}%` }} />
            </span>
            <span className="t-micro muted">
              {(kind || asText(root.mode)).includes('regression') ? 'model confidence' : 'probability'}
            </span>
          </div>
        )}
      </div>

      {predictorKind === 'decision' && Object.keys(report).length > 0 && <DecisionQuality report={report} />}

      {summary && (
        <div className="run-block">
          <span className="run-block__label">Clinical summary</span>
          <p className="t-small secondary">{summary}</p>
        </div>
      )}

      {domains.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Domains processed</span>
          <div className="row gap-1 wrap">
            {domains.map((domain) => (
              <Badge key={domain} outline mono>
                {domain}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {findings.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Key findings</span>
          <FindingList items={findings} />
        </div>
      )}

      {reasoning.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Reasoning chain</span>
          <ol className="run-bullets run-bullets--ordered">
            {reasoning.map((line, index) => (
              <li key={index} className="t-small secondary">
                {line}
              </li>
            ))}
          </ol>
        </div>
      )}

      {uncertainty.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Uncertainty</span>
          <div className="row gap-1 wrap">
            {uncertainty.map((item, index) => (
              <Badge key={index} tone="caution">
                {item}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {Object.keys(root).length > 0 && <NodeView node={root} depth={0} context={context} />}

      {Object.keys(root).length === 0 && flat.length > 0 && (
        <div className="stack gap-3">
          {flat.map((node, index) => (
            <NodeView key={asText(node.node_id) || index} node={node} depth={0} context={context} />
          ))}
        </div>
      )}

      {Object.keys(payload).length > 0 && (
        <Disclosure title="Full payload" subtitle="Everything the predictor returned, verbatim">
          <pre className="run-pre run-pre--tall">{safeJson(payload)}</pre>
        </Disclosure>
      )}
    </div>
  )
})

/** The service adds this to the payload; an older payload is read off the root mode. */
function deriveOutputKind(root: Record<string, unknown>, flat: Record<string, unknown>[]): string {
  const mode = asText(root.mode) || asText(flat[0]?.mode)
  if (mode.includes('classification')) return 'classification'
  if (mode.includes('regression')) return 'regression'
  return ''
}

/**
 * Units per node and output, from the task the run was given. `unit_by_output`
 * wins; a measurement scale's unit fills any output it leaves out.
 */
function unitsByNode(spec: Record<string, unknown>): Map<string, Record<string, string>> {
  const out = new Map<string, Record<string, string>>()
  const visit = (raw: unknown, depth: number) => {
    const node = asRecord(raw)
    const nodeId = asText(node.node_id)
    if (nodeId) {
      const units: Record<string, string> = {}
      for (const [output, scale] of Object.entries(asRecord(node.output_scales))) {
        const unit = asText(asRecord(scale).unit)
        if (unit) units[output] = unit
      }
      for (const [output, unit] of Object.entries(asRecord(node.unit_by_output))) {
        if (asText(unit)) units[output] = asText(unit)
      }
      out.set(nodeId, units)
    }
    // A malformed spec must not recurse forever.
    if (depth < 32) for (const child of asArray(node.children)) visit(child, depth + 1)
  }
  visit(spec.root, 0)
  return out
}

/** Full precision for a tooltip, without a float's trailing noise. */
function precise(value: number | null): string {
  if (value === null) return '-'
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(10)))
}

function MeterRow({
  label,
  value,
  warn,
  note,
}: {
  label: ReactNode
  value: number | null
  warn: boolean
  note?: string
}) {
  return (
    <div className="run-probrow">
      <span className="t-tiny truncate">{label}</span>
      <span className="run-meter" aria-hidden>
        <span
          className={clsx('run-meter__fill', warn && 'run-meter__fill--warn')}
          style={{ width: `${Math.round(Math.max(0, Math.min(1, value ?? 0)) * 100)}%` }}
        />
      </span>
      <span className="t-tiny muted tabular">{value === null ? (note ?? '-') : percent(value, 0)}</span>
    </div>
  )
}

/**
 * How far a decision model's answers can be trusted: whether it judged the
 * record sufficient, how much the answers moved between presentation orders,
 * how much of the record fit its state, and what the requests cost.
 */
function DecisionQuality({ report }: { report: Record<string, unknown> }) {
  const quality = asRecord(report.quality)
  const settings = asRecord(report.settings)
  const sufficiency = asNumber(quality.evidence_sufficiency)
  const meanInstability = asNumber(quality.mean_instability)
  const maxInstability = asNumber(quality.max_instability)
  const confidence = asNumber(quality.mean_confidence)
  const coverage = asNumber(quality.feature_coverage)
  const requests = asArray(report.requests)
  const cost = asNumber(report.cost_usd)
  const inputTokens = asNumber(report.input_tokens)
  const modelId = asText(report.model)
  const modelLabel = asText(report.model_label) || modelId
  const orders = asNumber(settings.choice_orders)
  const levels = asNumber(settings.score_levels)
  const refine = settings.regression_refine

  const unstable = (maxInstability ?? meanInstability ?? 0) > INSTABILITY_CAUTION

  return (
    <div className="run-block">
      <span className="run-block__label">Decision quality</span>
      <div className="stack gap-1">
        <MeterRow
          label={
            <Tooltip content="The model's own answer to whether the record holds enough evidence for this task.">
              <span>Evidence sufficiency</span>
            </Tooltip>
          }
          value={sufficiency}
          warn={sufficiency !== null && sufficiency < 0.5}
          note="not asked"
        />
        <MeterRow label="Mean confidence" value={confidence} warn={confidence !== null && confidence < 0.5} />
        <MeterRow
          label={
            <Tooltip content="Share of the record's features that fit into the decision model's state.">
              <span>Feature coverage</span>
            </Tooltip>
          }
          value={coverage}
          warn={coverage !== null && coverage < 0.999}
        />
      </div>
      <dl className="kv">
        <dt>
          <Tooltip content="How far the answers moved when the options were shown in another order, as a multiple of the threshold. Above 1 the Critic rejects.">
            <span>Instability across orders</span>
          </Tooltip>
        </dt>
        <dd className="tabular" style={unstable ? { color: 'var(--caution)' } : undefined}>
          {meanInstability !== null ? `mean ${meanInstability.toFixed(2)}` : '-'}
          {maxInstability !== null ? `, max ${maxInstability.toFixed(2)}` : ''}
          {' of threshold'}
        </dd>
        <dt>Requests</dt>
        <dd className="tabular">
          {requests.length}
          {inputTokens !== null ? `, ${formatTokens(inputTokens)} input tokens` : ''}
        </dd>
        <dt>Cost</dt>
        <dd className="tabular">{cost === null ? 'n/a' : `${usd(cost)} (input only)`}</dd>
        {modelLabel && (
          <>
            <dt>Model</dt>
            <dd>
              {modelLabel}
              {modelId && modelId !== modelLabel && <span className="t-tiny muted mono">{`  ${modelId}`}</span>}
            </dd>
          </>
        )}
        {(orders !== null || levels !== null) && (
          <>
            <dt>Questioning</dt>
            <dd className="t-tiny secondary">
              {[
                orders !== null ? `${orders} option order${orders === 1 ? '' : 's'}` : '',
                levels !== null ? `${levels} score levels` : '',
                typeof refine === 'boolean' ? (refine ? 'zoomed refinement on' : 'no refinement') : '',
              ]
                .filter(Boolean)
                .join(', ')}
            </dd>
          </>
        )}
      </dl>
    </div>
  )
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function NodeView({ node, depth, context }: { node: Record<string, unknown>; depth: number; context: NodeContext }) {
  const nodeId = asText(node.node_id) || 'root'
  const path = asText(node.path)
  const mode = asText(node.mode)
  const classification = asRecord(node.classification)
  const regression = asRecord(node.regression)
  const probabilities = asRecord(classification.probabilities)
  const values = asRecord(regression.values)
  const spread = asRecord(regression.uncertainty)
  const hasSpread = Object.keys(values).some((key) => Object.keys(asRecord(spread[key])).length > 0)
  const units = context.units.get(nodeId) ?? {}
  const details = asRecord(node.decision_details)
  const nodeStats = asRecord(context.perNode[nodeId])
  const unassessed = details.unassessed === true || nodeStats.unassessed === true
  const rawInstability = asNumber(details.instability) ?? asNumber(nodeStats.instability)
  const instabilityThreshold = asNumber(details.instability_threshold) ?? asNumber(nodeStats.threshold)
  const instability =
    asNumber(nodeStats.instability_ratio) ??
    (rawInstability !== null && instabilityThreshold ? rawInstability / instabilityThreshold : null)
  const sufficiency = asNumber(details.evidence_sufficiency)
  const confidenceScore = asNumber(node.confidence_score)
  const confidenceLevel = asText(node.confidence_level)
  const reasoning = asStringList(node.reasoning_chain)
  const evidenceFor = asStringList(node.supporting_evidence_for)
  const evidenceAgainst = asStringList(node.supporting_evidence_against)
  const uncertainty = asStringList(node.uncertainty_factors)
  const findings = asArray(node.key_findings)
  const children = asArray(node.children).map((row) => asRecord(row))

  return (
    <div className="run-node" style={{ marginLeft: depth * 14 }}>
      <header className="row gap-2 wrap between">
        <div className="row gap-2 wrap" style={{ minWidth: 0 }}>
          <span className="t-small semibold truncate">{path || nodeId}</span>
          {mode && <Badge tone="neutral">{titleCase(mode)}</Badge>}
          {confidenceLevel && <Badge outline>{titleCase(confidenceLevel)}</Badge>}
        </div>
        <span className="row gap-3">
          {unassessed && (
            <Tooltip content="Not every presentation order was answered, so stability could not be measured. The Critic treats this as unstable.">
              <span className="t-tiny tabular" style={{ color: 'var(--caution)' }}>
                stability not assessed
              </span>
            </Tooltip>
          )}
          {!unassessed && instability !== null && (
            <Tooltip
              content={`How much the answer moved when the options were reordered, as a multiple of its threshold${
                rawInstability !== null ? ` (raw ${rawInstability.toFixed(3)})` : ''
              }. 0 is identical in every order; above 1 the Critic rejects.`}
            >
              <span
                className="t-tiny tabular"
                style={{ color: instability > INSTABILITY_CAUTION ? 'var(--caution)' : 'var(--text-muted)' }}
              >
                instability {instability.toFixed(2)}
              </span>
            </Tooltip>
          )}
          {sufficiency !== null && (
            <span className="t-tiny muted tabular">evidence {sufficiency.toFixed(2)}</span>
          )}
          {confidenceScore !== null && (
            <span className="t-tiny muted tabular">confidence {confidenceScore.toFixed(2)}</span>
          )}
        </span>
      </header>

      {Object.keys(classification).length > 0 && (
        <div className="stack gap-2">
          <span className="t-tiny muted">
            Predicted label: <span className="semibold" style={{ color: 'var(--text)' }}>{asText(classification.predicted_label)}</span>
          </span>
          {Object.keys(probabilities).length > 0 && (
            <div className="stack gap-1">
              {Object.entries(probabilities).map(([key, raw]) => {
                const value = asNumber(raw) ?? 0
                return (
                  <div key={key} className="run-probrow">
                    <span className="t-tiny truncate">{key}</span>
                    <span className="run-meter" aria-hidden>
                      <span
                        className="run-meter__fill"
                        style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }}
                      />
                    </span>
                    <span className="t-tiny muted tabular">{percent(value, 1)}</span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {Object.keys(values).length > 0 && (
        <div className="run-tablewrap">
          <table className="table">
            <thead>
              <tr>
                <th>Output</th>
                <th className="num">Value</th>
                {hasSpread && (
                  <>
                    <th className="num">SD</th>
                    <th className="num">90% interval</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {Object.entries(values).map(([key, raw]) => {
                const value = asNumber(raw)
                const dist = asRecord(spread[key])
                const sd = asNumber(dist.sd)
                const q05 = asNumber(dist.q05)
                const q95 = asNumber(dist.q95)
                const unit = units[key] ?? ''
                const suffix = unit ? ` ${unit}` : ''
                const tip = (
                  <span className="stack gap-1 tabular">
                    <span>
                      {key}: {precise(value)}
                      {suffix}
                    </span>
                    {sd !== null && <span>SD {precise(sd)}</span>}
                    {asNumber(dist.median) !== null && <span>Median {precise(asNumber(dist.median))}</span>}
                    {asNumber(dist.q25) !== null && asNumber(dist.q75) !== null && (
                      <span>
                        50% interval {precise(asNumber(dist.q25))} to {precise(asNumber(dist.q75))}
                      </span>
                    )}
                    {q05 !== null && q95 !== null && (
                      <span>
                        90% interval {precise(q05)} to {precise(q95)}
                      </span>
                    )}
                  </span>
                )
                return (
                  <tr key={key}>
                    <td>
                      {key}
                      {unit && <span className="t-tiny muted"> ({unit})</span>}
                    </td>
                    <td className="num tabular">
                      <Tooltip content={tip}>
                        <span>{value !== null ? value.toFixed(3) : asText(raw)}</span>
                      </Tooltip>
                    </td>
                    {hasSpread && (
                      <>
                        <td className="num tabular">{sd !== null ? sd.toFixed(3) : '-'}</td>
                        <td className="num tabular">
                          {q05 !== null && q95 !== null ? `${q05.toFixed(3)} to ${q95.toFixed(3)}` : '-'}
                        </td>
                      </>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {findings.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Findings</span>
          <FindingList items={findings} />
        </div>
      )}

      {reasoning.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Reasoning chain</span>
          <ol className="run-bullets run-bullets--ordered">
            {reasoning.map((line, index) => (
              <li key={index} className="t-tiny secondary">
                {line}
              </li>
            ))}
          </ol>
        </div>
      )}

      {(evidenceFor.length > 0 || evidenceAgainst.length > 0) && (
        <div className="grid grid--2">
          {evidenceFor.length > 0 && (
            <div className="run-block">
              <span className="run-block__label">Evidence for</span>
              <ul className="run-bullets">
                {evidenceFor.map((line, index) => (
                  <li key={index} className="t-tiny secondary">
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {evidenceAgainst.length > 0 && (
            <div className="run-block">
              <span className="run-block__label">Evidence against</span>
              <ul className="run-bullets">
                {evidenceAgainst.map((line, index) => (
                  <li key={index} className="t-tiny secondary">
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {uncertainty.length > 0 && (
        <div className="run-block">
          <span className="run-block__label">Uncertainty</span>
          <div className="row gap-1 wrap">
            {uncertainty.map((item, index) => (
              <Badge key={index} tone="caution">
                {item}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {children.map((child, index) => (
        <NodeView key={asText(child.node_id) || index} node={child} depth={depth + 1} context={context} />
      ))}
    </div>
  )
}

/** Findings arrive either as plain sentences or as structured rows. */
function FindingList({ items }: { items: unknown[] }) {
  return (
    <ul className="run-bullets">
      {items.map((item, index) => {
        if (typeof item === 'string') {
          return (
            <li key={index} className="t-small secondary">
              {item}
            </li>
          )
        }
        const row = asRecord(item)
        const domain = asText(row.domain)
        const zScore = asNumber(row.z_score)
        return (
          <li key={index} className="t-small secondary">
            {domain && <span className="semibold">{domain}: </span>}
            {asText(row.finding)}
            {asText(row.direction) && <span className="t-tiny muted"> ({asText(row.direction).toLowerCase()}</span>}
            {asText(row.direction) && zScore !== null && <span className="t-tiny muted">, z {zScore.toFixed(2)}</span>}
            {asText(row.direction) && <span className="t-tiny muted">)</span>}
          </li>
        )
      })}
    </ul>
  )
}
