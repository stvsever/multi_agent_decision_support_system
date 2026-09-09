/**
 * The summary tab: what the run concluded, on what evidence, and what it could
 * not see. Absent sections are named rather than hidden, because a silent gap
 * in a clinical report reads as a finding of normality.
 */

import { AlertTriangle, CircleCheck, ShieldQuestion } from 'lucide-react'
import { Badge, Callout, Card } from '@/components/ui/primitives'
import { dateTime, duration, number, percent, tokens as formatTokens, titleCase, usd } from '@/lib/format'
import type { CostSummary } from '@/lib/types'
import { Bar, BulletList, Flag, GroupBox, KeyValue, NotRecorded, Stat } from './parts'
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

function NodeDetail({ node }: { node: PredictionNode }) {
  if (isClassificationMode(node.mode)) {
    if (node.probabilities.length === 0) {
      return <NotRecorded what="A class probability distribution" />
    }
    return (
      <div className="stack">
        {node.probabilities.map((entry) => (
          <Bar
            key={entry.label}
            label={entry.label}
            value={entry.value}
            lead={entry.label === node.predictedLabel}
          />
        ))}
      </div>
    )
  }

  if (isRegressionMode(node.mode)) {
    if (node.values.length === 0) return <NotRecorded what="A set of regression outputs" />
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

  return <NotRecorded what={`A prediction body for mode "${node.mode}"`} />
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

function Evidence({ view }: { view: ReportView }) {
  const hasAnything =
    view.findings.length > 0 ||
    view.reasoning.length > 0 ||
    view.evidenceFor.length > 0 ||
    view.evidenceAgainst.length > 0

  if (!hasAnything) {
    return (
      <Card title="Evidence">
        <NotRecorded what="Key findings, a reasoning chain, and supporting evidence" />
      </Card>
    )
  }

  return (
    <Card
      title="Evidence"
      subtitle={
        view.findingsSource === 'report'
          ? 'Key findings read from the structured report; the prediction tree carried none.'
          : undefined
      }
    >
      <div className="stack gap-5">
        {view.findings.length > 0 ? (
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
        ) : (
          <NotRecorded what="Key findings" />
        )}

        <div className="stack gap-2">
          <span className="eyebrow">Reasoning chain</span>
          {view.reasoning.length > 0 ? <BulletList items={view.reasoning} ordered /> : <NotRecorded what="A reasoning chain" />}
        </div>

        <div className="grid grid--2">
          <GroupBox title="Supporting evidence for">
            {view.evidenceFor.length > 0 ? <BulletList items={view.evidenceFor} /> : <NotRecorded what="Supporting evidence" />}
          </GroupBox>
          <GroupBox title="Evidence against">
            {view.evidenceAgainst.length > 0 ? (
              <BulletList items={view.evidenceAgainst} />
            ) : (
              <span className="t-small muted">Nothing was recorded against this conclusion.</span>
            )}
          </GroupBox>
        </div>

        {view.uncertainty.length > 0 && (
          <GroupBox title="Uncertainty factors">
            <BulletList items={view.uncertainty} />
          </GroupBox>
        )}
      </div>
    </Card>
  )
}

function CoverageSection({ view }: { view: ReportView }) {
  const { coverage } = view
  const missing = coverage.missingCount ?? coverage.missingFeatures.length
  const shown = coverage.missingFeatures.slice(0, MISSING_FEATURE_CAP)
  const overflow = coverage.missingFeatures.length - shown.length
  const nothingMissing = coverage.present && missing === 0

  return (
    <Card title="What the run could not see">
      <div className="stack gap-4">
        {!coverage.present && <NotRecorded what="A coverage summary" />}

        {coverage.invariantOk === false && (
          <Callout tone="caution" icon={<AlertTriangle size={15} />} title="Coverage invariant failed">
            The engine could not account for every input feature. Treat the completeness of this report as unverified.
          </Callout>
        )}

        {nothingMissing && (
          <Callout tone="positive" icon={<CircleCheck size={15} />} title="Every input feature was represented">
            {number(coverage.representedFeatures ?? coverage.allFeatures)} of{' '}
            {number(coverage.allFeatures)} features reached the predictor, and the coverage invariant held.
          </Callout>
        )}

        {coverage.present && (
          <div className="grid grid--3">
            <Stat label="Features in" value={number(coverage.allFeatures)} />
            <Stat label="Represented" value={number(coverage.representedFeatures)} />
            <Stat
              label="Missing"
              value={number(missing)}
              meta={coverage.invariantOk === null ? 'invariant not reported' : coverage.invariantOk ? 'invariant held' : 'invariant failed'}
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

        <div className="stack gap-2">
          <span className="eyebrow">Domain coverage of the input</span>
          {view.domainCoverage.length === 0 ? (
            <NotRecorded what="A per-domain coverage breakdown" />
          ) : (
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
          )}
        </div>
      </div>
    </Card>
  )
}

function CriticSection({ view }: { view: ReportView }) {
  const { evaluation } = view

  if (evaluation.source === 'none') {
    return (
      <Card title="Critic evaluation">
        <Callout tone="neutral" icon={<ShieldQuestion size={15} />} title="No evaluation was recorded">
          Neither the performance report nor the structured report carries a critic block for this run.
          {view.verdict ? ` The run finished with verdict ${view.verdict}.` : ''}
        </Callout>
      </Card>
    )
  }

  const partial = evaluation.source === 'report'

  return (
    <Card
      title="Critic evaluation"
      subtitle={partial ? 'Read from the structured report; the full critic block was not written.' : undefined}
      actions={<Badge tone={verdictTone(evaluation.verdict)}>{evaluation.verdict || 'No verdict'}</Badge>}
    >
      <div className="stack gap-4">
        {partial && (
          <Callout tone="neutral">
            Score breakdown, checklist detail, strengths, weaknesses, and improvement suggestions were not recorded for
            this run. Only the verdict, the verdict confidence, and the checklist tally are available.
          </Callout>
        )}

        <div className="grid grid--3">
          <Stat
            label="Composite score"
            value={evaluation.compositeScore === null ? 'not recorded' : percent(evaluation.compositeScore, 0)}
          />
          <Stat
            label="Confidence in verdict"
            value={
              evaluation.confidenceInVerdict === null ? 'not recorded' : percent(evaluation.confidenceInVerdict, 0)
            }
          />
          <Stat
            label="Checklist"
            value={
              evaluation.checklistTotal === null || evaluation.checklistPassed === null
                ? 'not recorded'
                : `${evaluation.checklistPassed} of ${evaluation.checklistTotal}`
            }
            meta="checks passed"
          />
        </div>

        {evaluation.summary ? <p className="t-small secondary">{evaluation.summary}</p> : null}

        {evaluation.scoreBreakdown.length > 0 && (
          <GroupBox title="Score breakdown">
            <div className="stack">
              {evaluation.scoreBreakdown.map((entry) => (
                <Bar key={entry.key} label={titleCase(entry.key)} value={entry.value} />
              ))}
            </div>
          </GroupBox>
        )}

        {evaluation.checklist.length > 0 && (
          <GroupBox title="Checklist">
            <div className="reports__flags">
              {evaluation.checklist
                .filter((entry) => entry.active)
                .map((entry) => (
                  <Flag key={entry.key} label={titleCase(entry.key)} state={entry.passed} />
                ))}
            </div>
            {evaluation.checklist.some((entry) => !entry.active) && (
              <span className="t-tiny muted">
                Checks not applicable to this task mode are omitted:{' '}
                {evaluation.checklist
                  .filter((entry) => !entry.active)
                  .map((entry) => titleCase(entry.key))
                  .join(', ')}
                .
              </span>
            )}
          </GroupBox>
        )}

        {(evaluation.strengths.length > 0 || evaluation.weaknesses.length > 0) && (
          <div className="grid grid--2">
            <GroupBox title="Strengths">
              {evaluation.strengths.length > 0 ? (
                <BulletList items={evaluation.strengths} />
              ) : (
                <span className="t-small muted">None listed.</span>
              )}
            </GroupBox>
            <GroupBox title="Weaknesses">
              {evaluation.weaknesses.length > 0 ? (
                <BulletList items={evaluation.weaknesses} />
              ) : (
                <span className="t-small muted">None listed.</span>
              )}
            </GroupBox>
          </div>
        )}

        {evaluation.suggestions.length > 0 && (
          <div className="stack gap-2">
            <span className="eyebrow">Improvement suggestions</span>
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
          <GroupBox title="Domains missed">
            <div className="reports__chips">
              {evaluation.domainsMissed.map((domain) => (
                <Badge key={domain} tone="caution" mono>
                  {domain}
                </Badge>
              ))}
            </div>
          </GroupBox>
        )}
      </div>
    </Card>
  )
}

export function SummaryTab({ view, cost }: { view: ReportView; cost: CostSummary | null | undefined }) {
  const taskOutputs = view.classLabels.length
    ? view.classLabels.join(' vs ')
    : view.regressionOutputs.length
      ? view.regressionOutputs
          .map((output) => (view.unitByOutput[output] ? `${output} (${view.unitByOutput[output]})` : output))
          .join(', ')
      : 'not specified'

  const costText = cost && cost.usd !== null && cost.usd !== undefined ? usd(cost.usd) : 'Pricing unavailable'

  return (
    <>
      <Card title={view.participantId} subtitle={view.taskName || view.targetCondition || 'Prediction task'}>
        <div className="stack gap-4">
          <KeyValue
            rows={[
              { label: 'Task mode', value: view.taskMode ? titleCase(view.taskMode) : 'not specified' },
              { label: view.classLabels.length ? 'Labels' : 'Outputs', value: taskOutputs },
              {
                label: 'Context',
                value:
                  view.targetCondition || view.controlCondition
                    ? `${view.targetCondition || 'unspecified target'} against ${view.controlCondition || 'unspecified comparator'}`
                    : 'not specified',
              },
              { label: 'Executed', value: view.timestamp ? dateTime(view.timestamp) : 'timestamp not recorded' },
              {
                label: 'Duration',
                value: view.durationSeconds === null ? 'not recorded' : duration(view.durationSeconds),
              },
            ]}
          />
          <Callout
            tone="neutral"
            title={
              view.iterations === null
                ? 'Iteration count not recorded'
                : `Iteration ${view.selectedIteration ?? '?'} of ${view.iterations} was selected`
            }
          >
            {view.selectionReason || 'No selection reason was recorded for this run.'}
          </Callout>
        </div>
      </Card>

      <Card>
        <div className="grid grid--4">
          <Stat
            label="Primary output"
            // A multivariate output is a sentence, not a word, so it steps down
            // a size rather than pushing the row out of rhythm.
            value={
              <span
                className={view.primaryOutput.length > 24 ? 'reports__stat-value--long' : undefined}
                title={view.primaryOutput}
              >
                {view.primaryOutput || 'not recorded'}
              </span>
            }
            meta={view.taskName}
          />
          <Stat
            label="Confidence"
            value={view.confidenceLevel || 'not recorded'}
            meta={view.rootConfidence === null ? 'no score' : percent(view.rootConfidence, 0)}
          />
          <Stat
            label="Critic verdict"
            value={
              <span style={{ color: view.verdict === 'SATISFACTORY' ? 'var(--positive)' : view.verdict === 'UNSATISFACTORY' ? 'var(--critical)' : 'var(--text)' }}>
                {view.verdict || 'not recorded'}
              </span>
            }
            meta={
              view.evaluation.compositeScore === null ? undefined : `score ${percent(view.evaluation.compositeScore, 0)}`
            }
          />
          <Stat
            label="Total tokens"
            value={view.totalTokens === null ? 'not recorded' : formatTokens(view.totalTokens)}
            meta={
              view.promptTokens === null
                ? undefined
                : `${formatTokens(view.promptTokens)} in, ${formatTokens(view.completionTokens)} out`
            }
          />
          <Stat label="Cost" value={costText} meta={cost ? `${cost.lines.length} model lines` : undefined} />
        </div>
      </Card>

      <Card
        title="Prediction detail"
        subtitle={view.isHierarchical ? 'Hierarchical task: the root and each child node' : undefined}
      >
        {view.root ? <NodeView node={view.root} root /> : <NotRecorded what="A prediction tree" />}
      </Card>

      <Evidence view={view} />

      {view.clinicalSummary ? (
        <Card title="Clinical summary">
          <p className="t-body secondary" style={{ maxWidth: '72ch', lineHeight: 1.7 }}>
            {view.clinicalSummary}
          </p>
        </Card>
      ) : null}

      <CoverageSection view={view} />

      <CriticSection view={view} />
    </>
  )
}
