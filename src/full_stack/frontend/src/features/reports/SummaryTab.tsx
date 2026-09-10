/**
 * The summary tab, composed as a document.
 *
 * A block with nothing in it is left out along with its heading: a heading over
 * "not recorded" is noise, and a stack of them buries the parts that do carry
 * a finding. What the run never wrote is stated once, at the end, so a gap is
 * still visible without being repeated on every section.
 */

import { AlertTriangle, CircleCheck } from 'lucide-react'
import type { ReactNode } from 'react'
import { Badge, Callout } from '@/components/ui/primitives'
import { dateTime, duration, number, percent, tokens as formatTokens, titleCase, usd } from '@/lib/format'
import type { CostSummary } from '@/lib/types'
import { Bar, BulletList, Flag, GroupBox, Omissions, Section, Stat } from './parts'
import {
  directionTone,
  isClassificationMode,
  isRegressionMode,
  nodeOutputText,
  VERDICT_TONE,
  type PredictionNode,
  type ReportView,
} from './model'

const MISSING_FEATURE_CAP = 40

function verdictTone(verdict: string) {
  return VERDICT_TONE[verdict] ?? 'neutral'
}

/* --- Prediction ----------------------------------------------------------- */

function NodeDetail({ node }: { node: PredictionNode }) {
  if (isClassificationMode(node.mode) && node.probabilities.length > 0) {
    return (
      <div className="stack">
        {node.probabilities.map((entry) => (
          <Bar key={entry.label} label={entry.label} value={entry.value} lead={entry.label === node.predictedLabel} />
        ))}
      </div>
    )
  }

  if (isRegressionMode(node.mode) && node.values.length > 0) {
    return (
      <table className="table">
        <thead>
          <tr>
            <th>Output</th>
            <th className="num">Value</th>
            <th>Unit</th>
          </tr>
        </thead>
        <tbody>
          {node.values.map((entry) => (
            <tr key={entry.output}>
              <td>{entry.output}</td>
              <td className="num">{entry.value === null ? 'not returned' : number(entry.value, 3)}</td>
              <td className="muted">{entry.unit || 'unitless'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return null
}

function NodeView({ node, root }: { node: PredictionNode; root?: boolean }) {
  return (
    <div className="reports__node">
      <div className="row between gap-3 wrap">
        <div className="stack" style={{ gap: 2, minWidth: 0 }}>
          <span className="row gap-2 wrap">
            <span className="semibold">{node.displayName || node.nodeId}</span>
            <Badge tone="neutral" mono>
              {titleCase(node.mode)}
            </Badge>
            {!root && <span className="t-micro muted mono">{node.path}</span>}
          </span>
          <span className="t-small secondary">{nodeOutputText(node)}</span>
        </div>
        <div className="row gap-2" style={{ flex: 'none' }}>
          {node.confidenceLevel ? <Badge tone="accent">{node.confidenceLevel}</Badge> : null}
          {node.confidenceScore !== null ? (
            <span className="row gap-2">
              <span className="reports__meter" aria-hidden>
                <span style={{ width: `${Math.max(0, Math.min(1, node.confidenceScore)) * 100}%` }} />
              </span>
              <span className="t-tiny tabular muted">{percent(node.confidenceScore, 0)}</span>
            </span>
          ) : null}
        </div>
      </div>

      <NodeDetail node={node} />

      {node.children.length > 0 && (
        <div className="reports__node-children">
          {node.children.map((child) => (
            <NodeView key={child.path || child.nodeId} node={child} />
          ))}
        </div>
      )}
    </div>
  )
}

/* --- Evidence ------------------------------------------------------------- */

function Evidence({ view }: { view: ReportView }) {
  const hasAnything =
    view.findings.length > 0 ||
    view.reasoning.length > 0 ||
    view.evidenceFor.length > 0 ||
    view.evidenceAgainst.length > 0 ||
    view.uncertainty.length > 0

  if (!hasAnything) return null

  return (
    <Section
      title="Evidence"
      note={
        view.findingsSource === 'report'
          ? 'Key findings read from the structured report; the prediction tree carried none.'
          : undefined
      }
    >
      {view.findings.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow">Key findings</span>
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: '20%' }}>Domain</th>
                <th>Finding</th>
                <th style={{ width: 140 }}>Direction</th>
                <th className="num" style={{ width: 70 }}>
                  z
                </th>
              </tr>
            </thead>
            <tbody>
              {view.findings.map((finding, index) => (
                <tr key={index}>
                  <td className="mono t-tiny">{finding.domain}</td>
                  <td>
                    {finding.finding}
                    {finding.relevance ? <div className="t-tiny muted">{finding.relevance}</div> : null}
                  </td>
                  <td>
                    {finding.direction ? (
                      <Badge tone={directionTone(finding.direction)}>{titleCase(finding.direction)}</Badge>
                    ) : (
                      <span className="muted t-tiny">not stated</span>
                    )}
                  </td>
                  <td className="num tabular">{finding.z === null ? '-' : finding.z.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {view.reasoning.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow">Reasoning chain</span>
          <BulletList items={view.reasoning} ordered />
        </div>
      )}

      {(view.evidenceFor.length > 0 || view.evidenceAgainst.length > 0) && (
        <div className="grid grid--2">
          {view.evidenceFor.length > 0 && (
            <GroupBox title="Supporting the conclusion">
              <BulletList items={view.evidenceFor} />
            </GroupBox>
          )}
          {view.evidenceAgainst.length > 0 && (
            <GroupBox title="Against the conclusion">
              <BulletList items={view.evidenceAgainst} />
            </GroupBox>
          )}
        </div>
      )}

      {view.uncertainty.length > 0 && (
        <GroupBox title="Uncertainty factors">
          <BulletList items={view.uncertainty} />
        </GroupBox>
      )}
    </Section>
  )
}

/* --- Coverage ------------------------------------------------------------- */

function CoverageSection({ view }: { view: ReportView }) {
  const { coverage } = view
  if (!coverage.present && view.domainCoverage.length === 0) return null

  const missing = coverage.missingCount ?? coverage.missingFeatures.length
  const shown = coverage.missingFeatures.slice(0, MISSING_FEATURE_CAP)
  const overflow = coverage.missingFeatures.length - shown.length

  return (
    <Section title="What the run could not see">
      {coverage.invariantOk === false && (
        <Callout tone="caution" icon={<AlertTriangle size={15} />} title="Coverage invariant failed">
          The engine could not account for every input feature. Treat the completeness of this report as unverified.
        </Callout>
      )}

      {coverage.present && missing === 0 && (
        <Callout tone="positive" icon={<CircleCheck size={15} />} title="Every input feature was represented">
          {number(coverage.representedFeatures ?? coverage.allFeatures)} of {number(coverage.allFeatures)} features
          reached the predictor, and the coverage invariant held.
        </Callout>
      )}

      {coverage.present && (
        <div className="grid grid--3">
          <Stat label="Features in" value={number(coverage.allFeatures)} />
          <Stat label="Represented" value={number(coverage.representedFeatures)} />
          <Stat
            label="Missing"
            value={number(missing)}
            meta={
              coverage.invariantOk === null
                ? undefined
                : coverage.invariantOk
                  ? 'invariant held'
                  : 'invariant failed'
            }
          />
        </div>
      )}

      {shown.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow">Missing features</span>
          <div className="reports__missing">
            {shown.map((feature) => (
              <Badge key={feature} mono outline>
                {feature}
              </Badge>
            ))}
            {overflow > 0 && <span className="t-tiny muted">and {number(overflow)} more</span>}
          </div>
        </div>
      )}

      {view.domainCoverage.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow">Domain coverage of the input</span>
          <table className="table">
            <thead>
              <tr>
                <th>Domain</th>
                <th className="num">Present</th>
                <th className="num">Total</th>
                <th className="num">Missing</th>
                <th style={{ width: '24%' }}>Coverage</th>
                <th className="num">Tokens</th>
              </tr>
            </thead>
            <tbody>
              {view.domainCoverage.map((row) => (
                <tr key={row.domain}>
                  <td className="mono t-tiny">
                    {row.domain}
                    {!row.available && (
                      <Badge tone="caution" style={{ marginLeft: 'var(--s-2)' }}>
                        unavailable
                      </Badge>
                    )}
                  </td>
                  <td className="num">{number(row.present)}</td>
                  <td className="num">{number(row.total)}</td>
                  <td className="num">{number(row.missing)}</td>
                  <td>
                    <Bar
                      label=""
                      value={(row.percentage ?? 0) / 100}
                      caption={row.percentage === null ? '-' : `${row.percentage.toFixed(1)}%`}
                    />
                  </td>
                  <td className="num">{formatTokens(row.tokens)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}

/* --- Critic --------------------------------------------------------------- */

function CriticSection({ view }: { view: ReportView }) {
  const { evaluation } = view
  if (evaluation.source === 'none') return null

  const partial = evaluation.source === 'report'
  const activeChecks = evaluation.checklist.filter((entry) => entry.active)

  return (
    <Section
      title="Critic evaluation"
      note={
        partial
          ? 'Read from the structured report. Only the verdict, its confidence, and the checklist tally were written.'
          : undefined
      }
      actions={<Badge tone={verdictTone(evaluation.verdict)}>{evaluation.verdict || 'No verdict'}</Badge>}
    >
      <div className="grid grid--3">
        {evaluation.compositeScore !== null && (
          <Stat label="Composite score" value={percent(evaluation.compositeScore, 0)} />
        )}
        {evaluation.confidenceInVerdict !== null && (
          <Stat label="Confidence in verdict" value={percent(evaluation.confidenceInVerdict, 0)} />
        )}
        {evaluation.checklistPassed !== null && evaluation.checklistTotal !== null && (
          <Stat
            label="Checklist"
            value={`${evaluation.checklistPassed} of ${evaluation.checklistTotal}`}
            meta="checks passed"
          />
        )}
      </div>

      {evaluation.summary ? <p className="reports__prose">{evaluation.summary}</p> : null}

      {evaluation.scoreBreakdown.length > 0 && (
        <GroupBox title="Score breakdown">
          <div className="stack">
            {evaluation.scoreBreakdown.map((entry) => (
              <Bar key={entry.key} label={titleCase(entry.key)} value={entry.value} />
            ))}
          </div>
        </GroupBox>
      )}

      {activeChecks.length > 0 && (
        <GroupBox title="Checklist">
          <div className="reports__flags">
            {activeChecks.map((entry) => (
              <Flag key={entry.key} label={titleCase(entry.key)} state={entry.passed} />
            ))}
          </div>
        </GroupBox>
      )}

      {(evaluation.strengths.length > 0 || evaluation.weaknesses.length > 0) && (
        <div className="grid grid--2">
          {evaluation.strengths.length > 0 && (
            <GroupBox title="Strengths">
              <BulletList items={evaluation.strengths} />
            </GroupBox>
          )}
          {evaluation.weaknesses.length > 0 && (
            <GroupBox title="Weaknesses">
              <BulletList items={evaluation.weaknesses} />
            </GroupBox>
          )}
        </div>
      )}

      {evaluation.suggestions.length > 0 && (
        <div className="stack gap-2">
          <span className="eyebrow">Improvements the critic asked for</span>
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: 96 }}>Priority</th>
                <th style={{ width: '30%' }}>Issue</th>
                <th>Suggestion</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.suggestions.map((entry, index) => (
                <tr key={index}>
                  <td>
                    <Badge
                      tone={entry.priority === 'HIGH' ? 'critical' : entry.priority === 'MEDIUM' ? 'caution' : 'neutral'}
                    >
                      {entry.priority || 'UNSET'}
                    </Badge>
                  </td>
                  <td>{entry.issue || '-'}</td>
                  <td className="secondary">{entry.suggestion || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {evaluation.domainsMissed.length > 0 && (
        <GroupBox title="Domains the critic found missing">
          <div className="reports__chips">
            {evaluation.domainsMissed.map((domain) => (
              <Badge key={domain} tone="caution" mono>
                {domain}
              </Badge>
            ))}
          </div>
        </GroupBox>
      )}
    </Section>
  )
}

/* --- The document --------------------------------------------------------- */

function metaLine(view: ReportView): { label: string; value: ReactNode }[] {
  const rows: { label: string; value: ReactNode }[] = []
  if (view.taskMode) rows.push({ label: 'Task', value: titleCase(view.taskMode) })
  if (view.classLabels.length) rows.push({ label: 'Labels', value: view.classLabels.join(' vs ') })
  else if (view.regressionOutputs.length) {
    rows.push({
      label: 'Outputs',
      value: view.regressionOutputs
        .map((output) => (view.unitByOutput[output] ? `${output} (${view.unitByOutput[output]})` : output))
        .join(', '),
    })
  }
  if (view.timestamp) rows.push({ label: 'Executed', value: dateTime(view.timestamp) })
  if (view.durationSeconds !== null) rows.push({ label: 'Took', value: duration(view.durationSeconds) })
  if (view.iterations !== null) {
    rows.push({
      label: 'Iterations',
      value:
        view.selectedIteration !== null ? `${view.selectedIteration} of ${view.iterations} selected` : String(view.iterations),
    })
  }
  return rows
}

/** One line saying what was asked, in the terms the task itself used. */
function lede(view: ReportView): string {
  const name = view.taskName || view.targetCondition || 'Prediction task'
  if (view.classLabels.length >= 2) {
    // A binary task is usually named after its positive label, and repeating it
    // reads as a stutter: "DEPRESSION: DEPRESSION against HEALTHY".
    const joined = view.classLabels.join(' against ')
    return view.classLabels[0] === name ? joined : `${name}: ${joined}`
  }
  if (view.targetCondition && view.controlCondition)
    return `${view.targetCondition} against ${view.controlCondition}`
  if (view.regressionOutputs.length && view.taskMode)
    return `${titleCase(view.taskMode)} of ${view.regressionOutputs.join(', ')}`
  return name
}

export function SummaryTab({ view, cost }: { view: ReportView; cost: CostSummary | null | undefined }) {
  const meta = metaLine(view)

  const stats: { label: string; value: ReactNode; meta?: ReactNode }[] = []
  if (view.primaryOutput) {
    stats.push({
      label: 'Primary output',
      value: (
        <span className={view.primaryOutput.length > 24 ? 'reports__stat-value--long' : undefined} title={view.primaryOutput}>
          {view.primaryOutput}
        </span>
      ),
      meta: view.taskName || undefined,
    })
  }
  if (view.confidenceLevel || view.rootConfidence !== null) {
    stats.push({
      label: 'Confidence',
      value: view.confidenceLevel || percent(view.rootConfidence, 0),
      meta: view.confidenceLevel && view.rootConfidence !== null ? percent(view.rootConfidence, 0) : undefined,
    })
  }
  if (view.verdict) {
    stats.push({
      label: 'Critic verdict',
      value: (
        <span
          style={{
            color:
              view.verdict === 'SATISFACTORY'
                ? 'var(--positive)'
                : view.verdict === 'UNSATISFACTORY'
                  ? 'var(--critical)'
                  : 'var(--text)',
          }}
        >
          {view.verdict}
        </span>
      ),
      meta:
        view.evaluation.compositeScore === null ? undefined : `score ${percent(view.evaluation.compositeScore, 0)}`,
    })
  }
  if (view.totalTokens !== null) {
    stats.push({
      label: 'Tokens',
      value: formatTokens(view.totalTokens),
      meta:
        view.promptTokens === null
          ? undefined
          : `${formatTokens(view.promptTokens)} in, ${formatTokens(view.completionTokens)} out`,
    })
  }
  // A zero with no priced lines behind it says nothing, so it is left out.
  if (cost && cost.usd !== null && cost.usd !== undefined && cost.lines.length > 0) {
    stats.push({
      label: 'Cost',
      value: usd(cost.usd),
      meta: `${cost.lines.length} ${cost.lines.length === 1 ? 'model' : 'models'}`,
    })
  }

  const omissions: string[] = []
  if (!view.root) omissions.push('a prediction tree')
  if (view.findingsSource === 'none') omissions.push('any key findings')
  if (!view.coverage.present) omissions.push('a coverage ledger')
  if (view.evaluation.source === 'none') omissions.push('a critic evaluation')

  return (
    <article className="reports__document">
      <header className="reports__doc-head">
        <span className="eyebrow">Prediction report</span>
        <h1 className="reports__doc-title">{view.participantId}</h1>
        <p className="reports__doc-lede">{lede(view)}</p>
        {meta.length > 0 && (
          <dl className="reports__doc-meta">
            {meta.map((row) => (
              <div key={row.label} className="reports__doc-meta-item">
                <dt>{row.label}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
        )}
      </header>

      {stats.length > 0 && (
        <div className="grid grid--4 reports__kpis">
          {stats.map((stat) => (
            <Stat key={stat.label} label={stat.label} value={stat.value} meta={stat.meta} />
          ))}
        </div>
      )}

      {view.selectionReason && <p className="reports__prose">{view.selectionReason}</p>}

      {view.root && (
        <Section
          title="Prediction"
          note={view.isHierarchical ? 'A hierarchical task: the root, then each child node.' : undefined}
        >
          <NodeView node={view.root} root />
        </Section>
      )}

      {view.clinicalSummary && (
        <Section title="Clinical summary">
          <p className="reports__prose">{view.clinicalSummary}</p>
        </Section>
      )}

      <Evidence view={view} />

      <CoverageSection view={view} />

      <CriticSection view={view} />

      <Omissions items={omissions} />
    </article>
  )
}
