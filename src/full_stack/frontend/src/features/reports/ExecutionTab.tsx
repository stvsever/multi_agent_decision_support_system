/**
 * How the run was carried out: plan, token ledger, spend, and the dataflow audit.
 *
 * This is where internal detail belongs, so it stays factual. It is not a place
 * to print the audit record's dotted keys verbatim: the numbers are grouped and
 * named in words here, and the record itself sits behind a disclosure for
 * anyone who wants to read it exactly as it was written.
 */

import type { ReactNode } from 'react'
import { Badge, Disclosure, EmptyState } from '@/components/ui/primitives'
import { number, percent, tokens as formatTokens, titleCase, usd, usdPerMillion } from '@/lib/format'
import type { CostSummary } from '@/lib/types'
import { Bar, Flag, GroupBox, KeyValue, Section, Stat } from './parts'
import { asBoolean, asDict, asNumber, asText, type Dict, type ReportView } from './model'

/** The audit writes schema names; these are the same checks in words. */
const ASSERTION_LABELS: Record<string, string> = {
  invariant_ok: 'The coverage invariant held',
  missing_feature_count_zero: 'No feature was dropped',
  chunk_evidence_matches_count: 'Every chunk reported evidence',
  processed_raw_flag_consistent: 'The processed raw flag matches the payload',
}

/**
 * The integrator records why it chunked or did not, as an enum token. These are
 * the four it writes, said in words; anything newer falls back to its own name.
 */
const CHUNKING_REASONS: Record<string, string> = {
  payload_fit_excluding_processed_raw: 'the payload fitted once the low priority raw block was left out',
  non_core_fit_under_budget: 'everything outside the core context already fitted inside the budget',
  no_chunkable_sections: 'there was nothing outside the core context to split',
  non_core_exceeds_budget: 'the material outside the core context was larger than one chunk',
}

function chunkingSkippedLine(reason: string): string {
  if (!reason) return 'Chunking was skipped.'
  const phrase = CHUNKING_REASONS[reason]
  if (phrase) return `Chunking was skipped because ${phrase}.`
  return `Chunking was skipped. The reason recorded was ${titleCase(reason)}.`
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/* --- Dataflow ------------------------------------------------------------- */

function DomainLeaves({ snapshot }: { snapshot: Dict }) {
  const unprocessed = asDict(snapshot.unprocessed_leaf_counts)
  const excluded = asDict(snapshot.excluded_leaf_counts)
  const domains = Array.from(new Set([...Object.keys(unprocessed), ...Object.keys(excluded)])).sort()
  if (domains.length === 0) return null

  return (
    <div className="stack gap-2">
      <span className="eyebrow">Leaves by domain</span>
      <table className="table">
        <thead>
          <tr>
            <th>Domain</th>
            <th className="num">Passed through raw</th>
            <th className="num">Summarised by a tool</th>
          </tr>
        </thead>
        <tbody>
          {domains.map((domain) => (
            <tr key={domain}>
              <td className="mono t-tiny">{domain}</td>
              <td className="num">{number(asNumber(unprocessed[domain]) ?? 0)}</td>
              <td className="num">{number(asNumber(excluded[domain]) ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Dataflow({ view }: { view: ReportView }) {
  const flow = view.dataflow
  if (Object.keys(flow).length === 0) return null

  const coverage = asDict(flow.coverage)
  const coverageSummary = asDict(coverage.summary)
  const chunking = asDict(flow.chunking)
  const contextFill = asDict(flow.context_fill)
  const payload = asDict(contextFill.predictor_payload_estimate)
  const snapshot = asDict(contextFill.coverage_snapshot)
  const embedding = asDict(contextFill.embedding_store)
  const assertions = asDict(flow.assertions)

  const overrides = Object.entries(asDict(flow.agent_instruction_flags))
    .filter(([, value]) => value === true)
    .map(([key]) => titleCase(key))

  const finalTokens = asNumber(payload.final_tokens)
  const limit = asNumber(payload.single_chunk_limit)
  const chunkCount = asNumber(chunking.predictor_chunk_count) ?? 0
  const twoPass = asBoolean(payload.chunked_two_pass_required)
  const skipped = asBoolean(chunking.chunking_skipped)
  const reason = asText(chunking.chunking_reason)
  const fallbackActive = asBoolean(embedding.fallback_active) === true

  // A row is printed only when the audit actually recorded its value, the way
  // every other block on this tab is guarded. A missing key says nothing worth
  // a line of its own.
  const identity: { label: string; value: ReactNode }[] = []
  const inputMode = asText(flow.predictor_input_mode)
  if (inputMode) identity.push({ label: 'Predictor input', value: <span className="mono t-tiny">{inputMode}</span> })
  const rootMode = asText(flow.prediction_task_root_mode)
  if (rootMode) identity.push({ label: 'Task mode', value: titleCase(rootMode) })
  const iteration = asNumber(flow.iteration)
  if (iteration !== null) identity.push({ label: 'Iteration', value: number(iteration) })
  if (Object.keys(asDict(flow.agent_instruction_flags)).length > 0) {
    identity.push({
      label: 'Instruction overrides',
      value: overrides.length ? overrides.join(', ') : 'none applied',
    })
  }

  return (
    <Section title="Dataflow" note="How the input reached the predictor, and which invariants held.">
      <KeyValue rows={identity} />

      {(finalTokens !== null || limit !== null) && (
        <GroupBox title="Payload">
          <div className="grid grid--3">
            {asNumber(payload.baseline_tokens) !== null && (
              <Stat label="Before enrichment" value={formatTokens(asNumber(payload.baseline_tokens))} meta="tokens" />
            )}
            {finalTokens !== null && <Stat label="Sent to the predictor" value={formatTokens(finalTokens)} meta="tokens" />}
            {limit !== null && <Stat label="Single chunk limit" value={formatTokens(limit)} meta="tokens" />}
          </div>
          {finalTokens !== null && limit !== null && limit > 0 && (
            <Bar label="" value={finalTokens / limit} caption={percent(finalTokens / limit, 0)} />
          )}
          {twoPass !== null && (
            <span className="t-small secondary">
              {twoPass
                ? 'The payload was large enough to need a two-pass chunked reading.'
                : 'The payload fitted in one pass, so no chunked reading was needed.'}
            </span>
          )}
        </GroupBox>
      )}

      {(chunkCount > 0 || skipped === true) && (
        <GroupBox title="Chunking">
          {chunkCount > 0 && (
            <div className="grid grid--3">
              <Stat label="Chunks sent" value={number(chunkCount)} />
              <Stat label="Chunks that cited evidence" value={number(asNumber(chunking.chunk_evidence_count) ?? 0)} />
            </div>
          )}
          {skipped === true && <span className="t-small secondary">{chunkingSkippedLine(reason)}</span>}
        </GroupBox>
      )}

      {Object.keys(coverageSummary).length > 0 && (
        <GroupBox title="Coverage">
          <div className="grid grid--4">
            <Stat label="Features in" value={number(asNumber(coverageSummary.all_feature_count))} />
            <Stat label="Through a tool" value={number(asNumber(coverageSummary.processed_feature_count))} />
            <Stat label="In the payload" value={number(asNumber(coverageSummary.represented_feature_count))} />
            <Stat
              label="Missing"
              value={number(asNumber(coverageSummary.missing_feature_count))}
              meta={
                asNumber(coverage.forced_raw_count)
                  ? `${number(asNumber(coverage.forced_raw_count))} forced into raw`
                  : undefined
              }
            />
          </div>
          <DomainLeaves snapshot={snapshot} />
        </GroupBox>
      )}

      {Object.keys(assertions).length > 0 && (
        <GroupBox title="Invariants">
          <div className="reports__flags">
            {Object.entries(assertions).map(([key, value]) => (
              <Flag
                key={key}
                label={ASSERTION_LABELS[key] ?? titleCase(key)}
                state={typeof value === 'boolean' ? value : null}
              />
            ))}
          </div>
        </GroupBox>
      )}

      {fallbackActive && (
        <span className="t-small secondary">
          The embedding store fell back to memory for this run.
          {asText(embedding.fallback_reason) ? ` Reason: ${asText(embedding.fallback_reason)}.` : ''}
        </span>
      )}

      <Disclosure title="Dataflow record" subtitle="The audit exactly as the engine wrote it">
        <pre className="reports__viewer">{safeJson(flow)}</pre>
      </Disclosure>
    </Section>
  )
}

/* --- The tab -------------------------------------------------------------- */

export function ExecutionTab({ view, cost }: { view: ReportView; cost: CostSummary | null | undefined }) {
  const ledgerTotal = view.tokenLines.reduce((sum, line) => sum + line.total, 0)
  const lines = cost?.lines ?? []
  const hasPlan = Boolean(view.plan.id) || view.plan.steps !== null || view.plan.domains.length > 0

  if (!hasPlan && view.tokenLines.length === 0 && lines.length === 0 && Object.keys(view.dataflow).length === 0) {
    return (
      <EmptyState
        title="No execution detail"
        body="This output directory holds no plan summary, token ledger, or dataflow audit. A run that ended before the executor finished leaves none."
      />
    )
  }

  return (
    <article className="reports__document">
      {hasPlan && (
        <Section title="Plan">
          <KeyValue
            rows={[
              ...(view.plan.steps === null ? [] : [{ label: 'Steps', value: number(view.plan.steps) }]),
              ...(view.plan.id ? [{ label: 'Plan id', value: <span className="mono t-tiny">{view.plan.id}</span> }] : []),
            ]}
          />
          {view.plan.domains.length > 0 && (
            <div className="stack gap-2">
              <span className="eyebrow">Priority domains</span>
              <div className="reports__chips">
                {view.plan.domains.map((domain) => (
                  <Badge key={domain} tone="accent" mono>
                    {domain}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </Section>
      )}

      {view.tokenLines.length > 0 && (
        <Section
          title="Token ledger"
          note="Every model call, grouped by the component that made it."
          actions={<span className="t-tiny muted tabular">{formatTokens(ledgerTotal)} total</span>}
        >
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
        </Section>
      )}

      {lines.length > 0 && (
        <Section
          title="Cost"
          actions={
            <span className="t-tiny muted">
              {cost && cost.usd !== null && cost.usd !== undefined ? usd(cost.usd) : 'pricing unavailable'}
            </span>
          }
        >
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
        </Section>
      )}

      <Dataflow view={view} />
    </article>
  )
}
