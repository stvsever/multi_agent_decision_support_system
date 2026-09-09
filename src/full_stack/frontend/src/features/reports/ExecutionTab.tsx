/** How the run was carried out: plan, token ledger, spend, and dataflow audit. */

import { Badge, Card } from '@/components/ui/primitives'
import { number, percent, tokens as formatTokens, titleCase, usd, usdPerMillion } from '@/lib/format'
import type { CostSummary } from '@/lib/types'
import { Bar, Flag, GroupBox, KeyValue, NotRecorded } from './parts'
import { asDict, type Dict, type ReportView } from './model'

/** Machine-readable audit blocks, rendered as labelled groups rather than raw JSON. */
function Structured({ value, depth = 0 }: { value: Dict; depth?: number }) {
  const scalars: { label: string; value: string }[] = []
  const flags: { label: string; state: boolean | null }[] = []
  const nested: [string, Dict][] = []
  const lists: [string, unknown[]][] = []

  for (const [key, raw] of Object.entries(value)) {
    // Deep keys are data (domain names, feature ids) rather than schema names,
    // so they keep their exact spelling.
    const label = depth >= 2 ? key : titleCase(key)
    if (typeof raw === 'boolean') flags.push({ label, state: raw })
    else if (raw === null || raw === undefined) scalars.push({ label, value: 'not recorded' })
    else if (typeof raw === 'number') scalars.push({ label, value: number(raw, Number.isInteger(raw) ? 0 : 3) })
    else if (typeof raw === 'string') scalars.push({ label, value: raw || 'not recorded' })
    else if (Array.isArray(raw)) lists.push([key, raw])
    else nested.push([key, asDict(raw)])
  }

  return (
    <div className="stack gap-3">
      {flags.length > 0 && (
        <div className="reports__flags">
          {flags.map((flag) => (
            <Flag key={flag.label} label={flag.label} state={flag.state} />
          ))}
        </div>
      )}
      {scalars.length > 0 && <KeyValue rows={scalars} />}
      {lists.map(([key, entries]) => (
        <div key={key} className="stack gap-1">
          <span className="t-tiny muted">{depth >= 2 ? key : titleCase(key)}</span>
          {entries.length === 0 ? (
            <span className="t-tiny faint">empty</span>
          ) : (
            <div className="reports__chips">
              {entries.slice(0, 24).map((entry, index) => (
                <Badge key={index} mono outline>
                  {typeof entry === 'object' ? JSON.stringify(entry) : String(entry)}
                </Badge>
              ))}
              {entries.length > 24 && <span className="t-tiny muted">and {entries.length - 24} more</span>}
            </div>
          )}
        </div>
      ))}
      {nested.map(([key, child]) =>
        depth >= 1 ? (
          <div key={key} className="stack gap-1">
            <span className="t-tiny muted">{depth >= 2 ? key : titleCase(key)}</span>
            <Structured value={child} depth={depth + 1} />
          </div>
        ) : (
          <GroupBox key={key} title={titleCase(key)}>
            <Structured value={child} depth={depth + 1} />
          </GroupBox>
        ),
      )}
    </div>
  )
}

function Dataflow({ view }: { view: ReportView }) {
  const flow = view.dataflow
  if (Object.keys(flow).length === 0) {
    return (
      <Card title="Dataflow">
        <NotRecorded what="A dataflow summary" />
      </Card>
    )
  }

  const coverage = asDict(flow.coverage)
  const chunking = asDict(flow.chunking)
  const contextFill = asDict(flow.context_fill)
  const assertions = asDict(flow.assertions)
  const instructionFlags = asDict(flow.agent_instruction_flags)
  const overrides = Object.entries(instructionFlags)
    .filter(([, value]) => value === true)
    .map(([key]) => key)

  return (
    <Card title="Dataflow" subtitle="How the input reached the predictor, and which invariants held">
      <div className="stack gap-4">
        <KeyValue
          rows={[
            { label: 'Iteration', value: number(typeof flow.iteration === 'number' ? flow.iteration : null) },
            {
              label: 'Root mode',
              value: typeof flow.prediction_task_root_mode === 'string' ? titleCase(flow.prediction_task_root_mode) : 'not recorded',
            },
            {
              label: 'Predictor input',
              value: typeof flow.predictor_input_mode === 'string' ? flow.predictor_input_mode : 'not recorded',
            },
            {
              label: 'Instruction overrides',
              value: overrides.length ? overrides.map(titleCase).join(', ') : 'none applied',
            },
          ]}
        />

        {Object.keys(assertions).length > 0 && (
          <GroupBox title="Assertions">
            <div className="reports__flags">
              {Object.entries(assertions).map(([key, value]) => (
                <Flag key={key} label={titleCase(key)} state={typeof value === 'boolean' ? value : null} />
              ))}
            </div>
          </GroupBox>
        )}

        {Object.keys(coverage).length > 0 && (
          <GroupBox title="Coverage">
            <Structured value={coverage} depth={1} />
          </GroupBox>
        )}

        {Object.keys(chunking).length > 0 && (
          <GroupBox title="Chunking">
            <Structured value={chunking} depth={1} />
          </GroupBox>
        )}

        {Object.keys(contextFill).length > 0 && (
          <GroupBox title="Context fill">
            <Structured value={contextFill} depth={1} />
          </GroupBox>
        )}
      </div>
    </Card>
  )
}

export function ExecutionTab({ view, cost }: { view: ReportView; cost: CostSummary | null | undefined }) {
  const ledgerTotal = view.tokenLines.reduce((sum, line) => sum + line.total, 0)
  const lines = cost?.lines ?? []

  return (
    <>
      <Card title="Plan">
        {view.plan.id || view.plan.steps !== null || view.plan.domains.length ? (
          <div className="stack gap-3">
            <KeyValue
              rows={[
                { label: 'Plan id', value: <span className="mono t-tiny">{view.plan.id || 'not recorded'}</span> },
                { label: 'Steps', value: view.plan.steps === null ? 'not recorded' : number(view.plan.steps) },
              ]}
            />
            <div className="stack gap-1">
              <span className="t-tiny muted">Priority domains</span>
              {view.plan.domains.length === 0 ? (
                <span className="t-tiny faint">none recorded</span>
              ) : (
                <div className="reports__chips">
                  {view.plan.domains.map((domain) => (
                    <Badge key={domain} tone="accent" mono>
                      {domain}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <NotRecorded what="A plan summary" />
        )}
      </Card>

      <Card
        title="Token ledger"
        subtitle="Grouped by the component that made the call"
        actions={<span className="t-tiny muted tabular">{formatTokens(ledgerTotal)} total</span>}
      >
        {view.tokenLines.length === 0 ? (
          <NotRecorded what="A per-call token ledger" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Component</th>
                <th className="num">Calls</th>
                <th className="num">Prompt</th>
                <th className="num">Completion</th>
                <th className="num">Total</th>
                <th style={{ width: '26%' }}>Share</th>
              </tr>
            </thead>
            <tbody>
              {view.tokenLines.map((line) => (
                <tr key={line.component}>
                  <td className="semibold">{titleCase(line.component)}</td>
                  <td className="num">{number(line.calls)}</td>
                  <td className="num">{formatTokens(line.prompt)}</td>
                  <td className="num">{formatTokens(line.completion)}</td>
                  <td className="num">{formatTokens(line.total)}</td>
                  <td>
                    <Bar
                      label=""
                      value={ledgerTotal > 0 ? line.total / ledgerTotal : 0}
                      caption={ledgerTotal > 0 ? percent(line.total / ledgerTotal, 0) : '-'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card
        title="Cost"
        actions={
          <span className="t-tiny muted">
            {cost && cost.usd !== null && cost.usd !== undefined ? usd(cost.usd) : 'Pricing unavailable'}
          </span>
        }
      >
        {lines.length === 0 ? (
          <span className="t-small muted">
            No priced usage lines are attached to this report. Cost appears once a run records usage against a model
            with published pricing.
          </span>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Prompt</th>
                <th className="num">Completion</th>
                <th className="num">Total</th>
                <th className="num">Rate in</th>
                <th className="num">Rate out</th>
                <th className="num">Spend</th>
              </tr>
            </thead>
            <tbody>
              {lines.map((line) => (
                <tr key={line.model}>
                  <td className="mono t-tiny">{line.model}</td>
                  <td className="num">{formatTokens(line.prompt_tokens)}</td>
                  <td className="num">{formatTokens(line.completion_tokens)}</td>
                  <td className="num">{formatTokens(line.total_tokens)}</td>
                  <td className="num muted">{usdPerMillion(line.prompt_usd_per_mtok)}</td>
                  <td className="num muted">{usdPerMillion(line.completion_usd_per_mtok)}</td>
                  <td className="num">{line.usd === null ? 'unpriced' : usd(line.usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Dataflow view={view} />
    </>
  )
}
