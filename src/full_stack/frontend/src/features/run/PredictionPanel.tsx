/**
 * The predictor's output.
 *
 * The payload shape follows the task: a classification node carries labelled
 * probabilities, a regression node carries values, and a hierarchical task
 * nests child nodes under the root. Everything is read defensively so a novel
 * payload still renders.
 */

import { memo } from 'react'
import { Badge, Disclosure, EmptyState } from '@/components/ui/primitives'
import { percent, titleCase, tokens as formatTokens } from '@/lib/format'
import { asArray, asNumber, asRecord, asStringList, asText } from './runUtils'

export interface PredictionPanelProps {
  prediction: Record<string, unknown> | null | undefined
}

export const PredictionPanel = memo(function PredictionPanel({ prediction }: PredictionPanelProps) {
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

  return (
    <div className="stack gap-4">
      <div className="run-verdict">
        <div className="stack gap-1" style={{ minWidth: 0 }}>
          <span className="stat__label">Predicted</span>
          <span className="t-h2 truncate">{label}</span>
          <div className="row gap-2 wrap">
            {kind && <Badge tone="neutral">{titleCase(kind)}</Badge>}
            {confidence && <Badge tone="info">{titleCase(confidence)} confidence</Badge>}
            {usedTokens !== null && <Badge outline>{formatTokens(usedTokens)} tokens</Badge>}
          </div>
        </div>
        {probability !== null && (
          <div className="run-prob">
            <span className="run-prob__value tabular">{percent(probability, 1)}</span>
            <span className="run-meter" aria-hidden>
              <span className="run-meter__fill" style={{ width: `${Math.round(Math.max(0, Math.min(1, probability)) * 100)}%` }} />
            </span>
            <span className="t-micro muted">probability</span>
          </div>
        )}
      </div>

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

      {Object.keys(root).length > 0 && <NodeView node={root} depth={0} />}

      {Object.keys(root).length === 0 && flat.length > 0 && (
        <div className="stack gap-3">
          {flat.map((node, index) => (
            <NodeView key={asText(node.node_id) || index} node={node} depth={0} />
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

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function NodeView({ node, depth }: { node: Record<string, unknown>; depth: number }) {
  const nodeId = asText(node.node_id) || 'root'
  const path = asText(node.path)
  const mode = asText(node.mode)
  const classification = asRecord(node.classification)
  const regression = asRecord(node.regression)
  const probabilities = asRecord(classification.probabilities)
  const values = asRecord(regression.values)
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
        {confidenceScore !== null && (
          <span className="t-tiny muted tabular">confidence {confidenceScore.toFixed(2)}</span>
        )}
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
        <table className="table">
          <thead>
            <tr>
              <th>Output</th>
              <th className="num">Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(values).map(([key, raw]) => (
              <tr key={key}>
                <td>{key}</td>
                <td className="num tabular">{asNumber(raw)?.toFixed(3) ?? asText(raw)}</td>
              </tr>
            ))}
          </tbody>
        </table>
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
        <NodeView key={asText(child.node_id) || index} node={child} depth={depth + 1} />
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
