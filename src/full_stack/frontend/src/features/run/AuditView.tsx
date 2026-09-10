/**
 * The structural audit console.
 *
 * An audit calls no model. It loads the participant's input files, assembles
 * the predictor payload and measures it, then checks a handful of invariants.
 * There is no plan graph and no agent traffic to watch, so this screen shows
 * what an audit actually proves instead of an empty live console.
 */

import clsx from 'clsx'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Check, CircleCheck, FlaskConical, RefreshCw, X } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, Callout, Disclosure, EmptyState, Progress, Spinner } from '@/components/ui/primitives'
import { api } from '@/lib/api'
import {
  bytes,
  duration as formatDuration,
  elapsedSince,
  tokens as formatTokens,
  titleCase,
} from '@/lib/format'
import { useNow } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { RunDetail } from '@/lib/types'
import { FailurePanel } from './FailurePanel'
import {
  assertionLabel,
  isActive,
  readAudit,
  sectionLabel,
  statusTone,
  summariseEvent,
  taskLine,
  taskSpecFromSummary,
  type AuditView as Audit,
} from './runUtils'

const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

/** Enough rows to see the shape of the payload without scrolling for a minute. */
const SECTION_PREVIEW = 12

export function AuditView({ detail }: { detail: RunDetail }) {
  const audit = useMemo(() => readAudit(detail), [detail])
  const running = isActive(detail.status)

  return (
    <div className="page run-audit">
      <AuditHeader detail={detail} running={running} />

      {detail.status === 'failed' && <FailurePanel detail={detail} />}

      <Callout
        tone="positive"
        icon={<CircleCheck size={15} />}
        title="No model was called, and nothing was spent"
      >
        A structural audit loads this participant's input files, builds the payload the predictor would
        receive, and measures it. It contacts no provider.
      </Callout>

      {running && <AuditProgress detail={detail} />}

      {!running && detail.status !== 'failed' && !audit.present && (
        <EmptyState
          icon={<FlaskConical size={22} />}
          title="This record carries no audit result"
          body={
            detail.status === 'cancelled'
              ? 'The audit was stopped before it wrote one. Start it again from the studio to produce a result.'
              : 'The run finished without writing one. Start the audit again from the studio to produce a fresh result.'
          }
        />
      )}

      {audit.present && (
        <>
          <InputCard detail={detail} audit={audit} />
          <PayloadCard audit={audit} />
          <CoverageCard audit={audit} />
          <SectionsCard audit={audit} />
          {audit.chunks.length > 0 && <ChunksCard audit={audit} />}
          <AssertionsCard audit={audit} />

          <Disclosure title="Raw audit record" subtitle="Everything the audit wrote, verbatim">
            <pre className="run-pre run-pre--tall">{safeJson(audit.raw)}</pre>
          </Disclosure>
        </>
      )}

      {!running && detail.events.length > 0 && (
        <Disclosure title="Progress log" subtitle={`${detail.events.length} events`}>
          <AuditEvents detail={detail} />
        </Disclosure>
      )}
    </div>
  )
}

/* --- Header --------------------------------------------------------------- */

function AuditHeader({ detail, running }: { detail: RunDetail; running: boolean }) {
  const navigate = useNavigate()
  const notify = useApp((s) => s.notify)
  const setTask = useApp((s) => s.setTask)
  const [busy, setBusy] = useState(false)
  const now = useNow(running)

  const seconds = detail.started_at
    ? detail.finished_at
      ? elapsedSince(detail.started_at, detail.finished_at)
      : Math.max(0, (now - Date.parse(detail.started_at)) / 1000)
    : null

  const cancel = useCallback(async () => {
    setBusy(true)
    try {
      await api.runs.cancel(detail.id)
      notify({
        tone: 'info',
        title: 'Cancelling audit',
        body: detail.participant_id,
      })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Could not cancel',
        body: (error as Error).message,
      })
    } finally {
      setBusy(false)
    }
  }, [detail.id, detail.participant_id, notify])

  const again = useCallback(() => {
    if (detail.task) setTask(taskSpecFromSummary(detail.task))
    notify({
      tone: 'info',
      title: 'Configuration carried over',
      body: 'Select the participant to audit or run the same task again.',
    })
    navigate('/studio')
  }, [detail.task, navigate, notify, setTask])

  return (
    <header className="run-head">
      <div className="run-head__top">
        <Link to="/runs" className="run-back">
          <ArrowLeft size={14} />
          <span className="t-tiny">Runs</span>
        </Link>
        <div className="grow" />
        {running && (
          <Button size="sm" variant="secondary" icon={<X size={14} />} loading={busy} onClick={cancel}>
            Cancel
          </Button>
        )}
        <Button size="sm" variant="ghost" icon={<RefreshCw size={14} />} onClick={again}>
          Run again
        </Button>
      </div>

      <div className="run-head__main">
        <div className="stack gap-2" style={{ minWidth: 0 }}>
          <div className="row gap-3 wrap">
            <h1 className="page__title truncate">{detail.participant_id}</h1>
            <Badge tone="info">Structural audit</Badge>
            <Badge tone={statusTone(detail.status)}>{STATUS_LABEL[detail.status] ?? detail.status}</Badge>
          </div>
          <span className="t-small muted truncate">{taskLine(detail.task)}</span>
        </div>

        <div className="run-head__stats">
          <div className="stat">
            <span className="stat__label">{detail.finished_at ? 'Duration' : 'Elapsed'}</span>
            <span className="stat__value tabular">{formatDuration(seconds)}</span>
            <span className="stat__meta">offline check</span>
          </div>
          <div className="stat">
            <span className="stat__label">Spend</span>
            <span className="stat__value tabular">$0.00</span>
            <span className="stat__meta">no provider calls</span>
          </div>
        </div>
      </div>
    </header>
  )
}

/* --- While it runs -------------------------------------------------------- */

function AuditProgress({ detail }: { detail: RunDetail }) {
  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">Working</span>
          <span className="t-tiny muted">{detail.state?.status || 'Loading the participant record'}</span>
        </div>
        <Spinner />
      </header>
      <div className="card__body stack gap-3">
        <Progress indeterminate />
        {detail.events.length > 0 && <AuditEvents detail={detail} />}
      </div>
    </section>
  )
}

function AuditEvents({ detail }: { detail: RunDetail }) {
  return (
    <ol className="run-events__list">
      {detail.events.map((event) => (
        <li key={event.id} className="run-event">
          <span className="run-event__time mono tabular">{event.time}</span>
          <span className="run-event__summary t-tiny secondary">{summariseEvent(event) || event.type}</span>
        </li>
      ))}
    </ol>
  )
}

/* --- The input the audit was pointed at ----------------------------------- */

function InputCard({ detail, audit }: { detail: RunDetail; audit: Audit }) {
  const participant = useQuery({
    queryKey: ['participant', detail.participant_dir],
    queryFn: () => api.datasets.participant(detail.participant_dir),
    enabled: Boolean(detail.participant_dir),
    staleTime: 60_000,
    retry: false,
  })

  const files = participant.data?.files ?? []

  // The audit record carries no file list of its own, so this table is a live
  // read of the directory. It is labelled as such rather than presented as the
  // set of files the audit actually opened.
  const noFiles = !detail.participant_dir
    ? 'This record does not name a participant directory.'
    : participant.isLoading
      ? 'Reading the participant directory.'
      : participant.isError
        ? 'The participant directory could not be read. It may have moved since the audit ran.'
        : 'The service read the participant directory and found no files in it.'

  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1, minWidth: 0 }}>
          <span className="card__title">Participant input</span>
          <span className="t-tiny muted truncate" title={detail.participant_dir}>
            {detail.participant_dir}
          </span>
        </div>
        {audit.taskMode && <Badge outline>{titleCase(audit.taskMode)}</Badge>}
      </header>
      <div className="card__body stack gap-3">
        {files.length > 0 ? (
          <>
            <span className="t-tiny muted">
              The participant record as it stands now. The audit wrote no file list of its own, so anything
              added or changed since it ran shows here too.
            </span>
            <div className="run-tablewrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th className="num">Size</th>
                    <th>State</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((file) => (
                    <tr key={file.key}>
                      <td className="mono t-tiny">{file.file}</td>
                      <td className="num tabular">{file.present ? bytes(file.size ?? 0) : '-'}</td>
                      <td>
                        {file.valid ? (
                          <span className="t-tiny muted">usable</span>
                        ) : (
                          <span className="t-tiny" style={{ color: 'var(--critical)' }}>
                            {file.issue || 'unusable'}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <span className="t-tiny muted">{noFiles}</span>
        )}

        {audit.nodeCount !== null && audit.nodeCount > 0 && (
          <span className="t-tiny muted">
            The task tree carries {audit.nodeCount} {audit.nodeCount === 1 ? 'node' : 'nodes'}
            {audit.targetCondition ? `, targeting ${audit.targetCondition}` : ''}
            {audit.controlCondition ? ` against ${audit.controlCondition}` : ''}.
          </span>
        )}
      </div>
    </section>
  )
}

/* --- Payload against the budget ------------------------------------------- */

function PayloadCard({ audit }: { audit: Audit }) {
  const payload = audit.payloadTokens
  const budget = audit.chunkBudget
  const share = payload !== null && budget ? payload / budget : null
  const chunks = audit.chunkCount ?? 0
  const chunked = chunks > 1

  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">Predictor payload</span>
          <span className="t-tiny muted">What the predictor would have been handed</span>
        </div>
        {audit.inputMode && (
          <Badge tone="neutral" mono>
            {audit.inputMode}
          </Badge>
        )}
      </header>
      <div className="card__body stack gap-4">
        <p className="t-body secondary" style={{ maxWidth: '68ch' }}>
          {payload === null || budget === null
            ? 'The audit did not record a payload size.'
            : `The payload is ${formatTokens(payload)} tokens against a ${formatTokens(budget)} token chunk budget.`}{' '}
          {chunks === 0
            ? 'Nothing needed splitting.'
            : chunked
              ? `It would be sent as ${chunks} chunks.`
              : 'It fits in a single chunk.'}
        </p>

        {share !== null && (
          <div className="stack gap-1">
            <span className="run-budget" aria-hidden>
              <span
                className={clsx('run-budget__fill', share > 1 && 'run-budget__fill--over')}
                style={{ width: `${Math.round(Math.min(1, share) * 100)}%` }}
              />
            </span>
            <div className="row between">
              <span className="t-micro faint tabular">0</span>
              <span className="t-micro muted tabular">{Math.round(share * 100)}% of the chunk budget</span>
              <span className="t-micro faint tabular">{formatTokens(budget)}</span>
            </div>
          </div>
        )}

        <div className="run-costtotals">
          <div className="stat">
            <span className="stat__label">Payload</span>
            <span className="stat__value tabular">{payload === null ? '-' : formatTokens(payload)}</span>
            <span className="stat__meta">tokens</span>
          </div>
          <div className="stat">
            <span className="stat__label">Chunk budget</span>
            <span className="stat__value tabular">{budget === null ? '-' : formatTokens(budget)}</span>
            <span className="stat__meta">tokens per chunk</span>
          </div>
          <div className="stat">
            <span className="stat__label">Chunks</span>
            <span className="stat__value tabular">{audit.chunkCount ?? '-'}</span>
            <span className="stat__meta">{chunked ? 'two-pass reading' : 'single pass'}</span>
          </div>
        </div>
      </div>
    </section>
  )
}

/* --- Coverage ------------------------------------------------------------- */

function CoverageCard({ audit }: { audit: Audit }) {
  const { coverage } = audit
  if (!coverage.present) return null

  const missing = coverage.missing
  const forced = coverage.forcedRaw ?? 0

  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">Coverage ledger</span>
          <span className="t-tiny muted">Every input feature, and where it ended up</span>
        </div>
      </header>
      <div className="card__body stack gap-4">
        {missing === 0 && (
          <Callout
            tone="positive"
            icon={<CircleCheck size={15} />}
            title="Every input feature reached the payload"
          >
            {coverage.all === null
              ? 'The ledger accounted for all of them.'
              : `All ${coverage.all} features are accounted for.`}
          </Callout>
        )}
        {missing !== null && missing > 0 && (
          <Callout tone="critical" title={`${missing} features never reached the payload`}>
            The predictor would be reading an incomplete record. Treat any run on this input as unverified.
          </Callout>
        )}

        <div className="run-costtotals">
          <div className="stat">
            <span className="stat__label">Features in</span>
            <span className="stat__value tabular">{coverage.all ?? '-'}</span>
          </div>
          <div className="stat">
            <span className="stat__label">Processed</span>
            <span className="stat__value tabular">{coverage.processed ?? '-'}</span>
            <span className="stat__meta">through a tool</span>
          </div>
          <div className="stat">
            <span className="stat__label">Covered</span>
            <span className="stat__value tabular">{coverage.covered ?? '-'}</span>
            <span className="stat__meta">present in the payload</span>
          </div>
          <div className="stat">
            <span className="stat__label">Missing</span>
            <span
              className="stat__value tabular"
              style={missing !== null && missing > 0 ? { color: 'var(--critical)' } : undefined}
            >
              {coverage.missing ?? '-'}
            </span>
            {forced > 0 && <span className="stat__meta">{forced} forced into raw</span>}
          </div>
        </div>

        {coverage.processed === 0 && (
          <span className="t-tiny muted">
            An audit dispatches no tools, so nothing is summarised and every feature is carried through as a raw
            value.
          </span>
        )}
      </div>
    </section>
  )
}

/* --- Sections and chunks -------------------------------------------------- */

function SectionsCard({ audit }: { audit: Audit }) {
  const [all, setAll] = useState(false)
  if (audit.sections.length === 0) return null

  const total = audit.sections.reduce((sum, section) => sum + section.tokens, 0)
  const rows = all ? audit.sections : audit.sections.slice(0, SECTION_PREVIEW)
  const hidden = audit.sections.length - rows.length

  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">Sections by size</span>
          <span className="t-tiny muted">
            {audit.sections.length} sections, {formatTokens(total)} tokens before assembly
          </span>
        </div>
      </header>
      <div className="card__body">
        <div className="run-tablewrap">
          <table className="table">
            <thead>
              <tr>
                <th>Section</th>
                <th className="num">Features</th>
                <th className="num">Tokens</th>
                <th style={{ width: '28%' }}>Share</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((section) => {
                const label = sectionLabel(section.name)
                const share = total > 0 ? section.tokens / total : 0
                return (
                  <tr key={section.name}>
                    <td>
                      <span className="row gap-2" style={{ minWidth: 0 }}>
                        <span className="truncate" title={section.name}>
                          {label.title}
                        </span>
                        {label.part && <span className="t-micro faint">{label.part}</span>}
                      </span>
                    </td>
                    <td className="num tabular muted">{section.featureKeys || '-'}</td>
                    <td className="num tabular">{formatTokens(section.tokens)}</td>
                    <td>
                      <span className="run-meter" aria-hidden>
                        <span className="run-meter__fill" style={{ width: `${Math.round(share * 100)}%` }} />
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {hidden > 0 && (
          <div className="row" style={{ paddingTop: 'var(--s-3)' }}>
            <Button size="sm" variant="ghost" onClick={() => setAll(true)}>
              Show the other {hidden}
            </Button>
          </div>
        )}
      </div>
    </section>
  )
}

function ChunksCard({ audit }: { audit: Audit }) {
  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">How it would be chunked</span>
          <span className="t-tiny muted">The order the predictor would read the payload in</span>
        </div>
      </header>
      <div className="card__body stack gap-2">
        {audit.chunks.map((chunk) => (
          <div key={chunk.index} className="run-chunk">
            <span className="run-chunk__index tabular">{chunk.index}</span>
            <div className="stack gap-1" style={{ minWidth: 0 }}>
              <span className="t-small semibold tabular">{formatTokens(chunk.tokens)} tokens</span>
              <div className="row gap-1 wrap">
                {chunk.sections.map((name) => {
                  const label = sectionLabel(name)
                  return (
                    <Badge key={name} outline>
                      {label.title}
                      {label.part ? ` ${label.part}` : ''}
                    </Badge>
                  )
                })}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

/* --- Assertions ----------------------------------------------------------- */

function AssertionsCard({ audit }: { audit: Audit }) {
  if (audit.assertions.length === 0) return null
  const held = audit.assertions.filter((entry) => entry.ok).length

  return (
    <section className="card">
      <header className="card__header">
        <div className="stack" style={{ gap: 1 }}>
          <span className="card__title">Invariants</span>
          <span className="t-tiny muted">Checked against the payload the audit just built</span>
        </div>
        <Badge tone={audit.assertionsOk === false ? 'critical' : 'positive'}>
          {audit.assertionsOk === false ? `${held} of ${audit.assertions.length} held` : 'all held'}
        </Badge>
      </header>
      <div className="card__body">
        <ul className="run-checks">
          {audit.assertions.map((entry) => (
            <li
              key={entry.key}
              className={clsx('run-check', entry.ok ? 'run-check--pass' : 'run-check--fail')}
            >
              <span className="run-check__icon" aria-hidden>
                {entry.ok ? <Check size={12} /> : <X size={12} />}
              </span>
              <span className="t-tiny">{assertionLabel(entry.key)}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}
