/**
 * Run history.
 *
 * Every run the service knows about, filterable and sortable, with the live
 * ones kept current by a four-second poll.
 */

import clsx from 'clsx'
import { useQueryClient } from '@tanstack/react-query'
import { Activity, FileText, Search, Sparkles, Trash2, X } from 'lucide-react'
import { memo, useCallback, useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, EmptyState, Input, Progress, Segmented, Skeleton, Tooltip } from '@/components/ui/primitives'
import { api } from '@/lib/api'
import { duration as formatDuration, elapsedSince, relativeTime, tokens as formatTokens, usd } from '@/lib/format'
import { queryKeys, useNow, useRuns } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { RunStatus, RunSummary } from '@/lib/types'
import { isActive, isTerminal, statusTone, taskModeLabel, verdictTone } from './runUtils'
import './run.css'

type SortKey = 'newest' | 'longest' | 'costliest'

const STATUS_ORDER: RunStatus[] = ['running', 'queued', 'cancelling', 'succeeded', 'failed', 'cancelled']

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function runSeconds(run: RunSummary): number | null {
  if (!run.started_at) return null
  return elapsedSince(run.started_at, run.finished_at)
}

export function RunsPage() {
  const { data, isLoading } = useRuns(true)
  const runs = useMemo(() => data?.runs ?? [], [data])
  const [status, setStatus] = useState<RunStatus | 'all'>('all')
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<SortKey>('newest')

  const anyActive = runs.some((run) => isActive(run.status))
  const now = useNow(anyActive)

  const counts = useMemo(() => {
    const table: Record<string, number> = {}
    for (const run of runs) table[run.status] = (table[run.status] ?? 0) + 1
    return table
  }, [runs])

  const spend = useMemo(
    () => runs.reduce((total, run) => total + (run.cost?.usd ?? 0), 0),
    [runs],
  )
  const spentTokens = useMemo(
    () => runs.reduce((total, run) => total + (run.cost?.total_tokens ?? 0), 0),
    [runs],
  )

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    const filtered = runs.filter((run) => {
      if (status !== 'all' && run.status !== status) return false
      if (!needle) return true
      return (
        run.participant_id.toLowerCase().includes(needle) ||
        run.label.toLowerCase().includes(needle) ||
        run.id.toLowerCase().includes(needle)
      )
    })
    const sorted = [...filtered]
    if (sort === 'longest') {
      sorted.sort((a, b) => (runSeconds(b) ?? 0) - (runSeconds(a) ?? 0))
    } else if (sort === 'costliest') {
      sorted.sort((a, b) => (b.cost?.usd ?? 0) - (a.cost?.usd ?? 0))
    } else {
      sorted.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
    }
    return sorted
  }, [runs, status, search, sort])

  return (
    <div className="page">
      <header className="page__head">
        <div className="stack gap-1">
          <h1 className="page__title">Runs</h1>
          <p className="page__lede">
            Every pipeline execution, live and past. Open one to watch the agents work.
          </p>
        </div>
        <div className="row gap-5 wrap">
          <div className="stat">
            <span className="stat__label">Total spend</span>
            <span className="stat__value">{usd(spend)}</span>
            <span className="stat__meta">{formatTokens(spentTokens)} tokens across {runs.length} runs</span>
          </div>
        </div>
      </header>

      <div className="run-toolbar">
        <div className="run-chips">
          <FilterChip active={status === 'all'} onClick={() => setStatus('all')} count={runs.length}>
            All
          </FilterChip>
          {STATUS_ORDER.filter((key) => counts[key]).map((key) => (
            <FilterChip
              key={key}
              active={status === key}
              onClick={() => setStatus(status === key ? 'all' : key)}
              count={counts[key]}
              dot={key}
            >
              {STATUS_LABEL[key]}
            </FilterChip>
          ))}
        </div>
        <div className="grow" />
        <div className="run-search">
          <Search size={14} className="run-search__icon" />
          <Input
            value={search}
            placeholder="Participant or label"
            aria-label="Search runs"
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <Segmented<SortKey>
          value={sort}
          onChange={setSort}
          options={[
            { value: 'newest', label: 'Newest' },
            { value: 'longest', label: 'Longest' },
            { value: 'costliest', label: 'Costliest' },
          ]}
        />
      </div>

      {isLoading && runs.length === 0 ? (
        <div className="stack gap-2">
          <Skeleton height={62} radius={12} />
          <Skeleton height={62} radius={12} />
          <Skeleton height={62} radius={12} />
        </div>
      ) : visible.length === 0 ? (
        <EmptyState
          icon={<Activity size={22} />}
          title={runs.length === 0 ? 'No runs yet' : 'Nothing matches those filters'}
          body={
            runs.length === 0
              ? 'Compose a run in the studio: pick a participant, describe the prediction, and launch.'
              : 'Clear the search or pick another status to see the rest of the history.'
          }
          action={
            runs.length === 0 ? (
              <Link to="/studio">
                <Button variant="primary" icon={<Sparkles size={14} />}>
                  Configure a run
                </Button>
              </Link>
            ) : (
              <Button
                onClick={() => {
                  setSearch('')
                  setStatus('all')
                }}
              >
                Reset filters
              </Button>
            )
          }
        />
      ) : (
        <ul className="run-list">
          {visible.map((run) => (
            <RunRow key={run.id} run={run} now={isActive(run.status) ? now : 0} />
          ))}
        </ul>
      )}
    </div>
  )
}

function FilterChip({
  active,
  count,
  dot,
  onClick,
  children,
}: {
  active: boolean
  count?: number
  dot?: RunStatus
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button type="button" className="run-chip" aria-pressed={active} onClick={onClick}>
      {dot && <span className={clsx('run-dot', `run-dot--${dot}`)} aria-hidden />}
      <span>{children}</span>
      {count !== undefined && <span className="run-chip__count tabular">{count}</span>}
    </button>
  )
}

const RunRow = memo(function RunRow({ run, now }: { run: RunSummary; now: number }) {
  const client = useQueryClient()
  const navigate = useNavigate()
  const notify = useApp((s) => s.notify)
  const [busy, setBusy] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const active = isActive(run.status)
  // `now` ticks only for live rows, which is what keeps the elapsed time moving.
  const seconds = run.started_at
    ? run.finished_at
      ? elapsedSince(run.started_at, run.finished_at)
      : Math.max(0, ((now || Date.now()) - Date.parse(run.started_at)) / 1000)
    : null

  const cancel = useCallback(async () => {
    setBusy(true)
    try {
      await api.runs.cancel(run.id)
      notify({ tone: 'info', title: 'Cancelling run', body: run.participant_id })
    } catch (error) {
      notify({ tone: 'critical', title: 'Could not cancel', body: (error as Error).message })
    } finally {
      setBusy(false)
      client.invalidateQueries({ queryKey: queryKeys.runs })
    }
  }, [client, notify, run.id, run.participant_id])

  const remove = useCallback(async () => {
    setBusy(true)
    try {
      await api.runs.remove(run.id)
      notify({ tone: 'info', title: 'Run deleted', body: run.participant_id })
    } catch (error) {
      notify({ tone: 'critical', title: 'Could not delete', body: (error as Error).message })
    } finally {
      setBusy(false)
      setConfirmDelete(false)
      client.invalidateQueries({ queryKey: queryKeys.runs })
    }
  }, [client, notify, run.id, run.participant_id])

  return (
    <li className={clsx('run-row', active && 'run-row--live')}>
      <Link className="run-row__main" to={`/runs/${run.id}`}>
        <span className={clsx('run-dot', 'run-dot--lg', `run-dot--${run.status}`)} aria-hidden />

        <span className="run-row__identity">
          <span className="row gap-2" style={{ minWidth: 0 }}>
            <span className="semibold truncate">{run.participant_id}</span>
            <Badge tone="neutral">{taskModeLabel(run.task)}</Badge>
            {run.audit && <Badge tone="info">Audit</Badge>}
            {run.verdict && <Badge tone={verdictTone(run.verdict)}>{run.verdict}</Badge>}
          </span>
          <span className="t-tiny muted truncate">
            {run.label !== run.participant_id ? `${run.label} - ` : ''}
            {relativeTime(run.created_at)}
          </span>
        </span>

        <span className="run-row__progress">
          {active ? (
            <>
              <Progress value={run.progress} max={Math.max(1, run.max_steps)} />
              <span className="t-micro muted tabular">
                {run.progress} of {run.max_steps} steps
              </span>
            </>
          ) : (
            <Badge tone={statusTone(run.status)}>{STATUS_LABEL[run.status]}</Badge>
          )}
        </span>

        <span className="run-row__metrics">
          <span className="run-metric">
            <span className="run-metric__value tabular">{formatDuration(seconds)}</span>
            <span className="run-metric__label">{run.finished_at ? 'duration' : 'elapsed'}</span>
          </span>
          <span className="run-metric">
            <span className="run-metric__value tabular">{formatTokens(run.cost?.total_tokens ?? 0)}</span>
            <span className="run-metric__label">tokens</span>
          </span>
          <span className="run-metric">
            <span className="run-metric__value tabular">{usd(run.cost?.usd)}</span>
            <span className="run-metric__label">of ~{usd(run.estimate?.usd)}</span>
          </span>
        </span>
      </Link>

      <div className="run-row__actions">
        {active && (
          <Tooltip content="Cancel this run">
            <Button
              size="sm"
              variant="ghost"
              iconOnly
              icon={<X size={14} />}
              loading={busy}
              onClick={cancel}
              aria-label="Cancel run"
            />
          </Tooltip>
        )}
        {run.status === 'succeeded' && (
          <Tooltip content="Open the report">
            <Button
              size="sm"
              variant="ghost"
              iconOnly
              icon={<FileText size={14} />}
              onClick={() => navigate(`/reports/${encodeURIComponent(run.participant_id)}?run=${run.id}`)}
              aria-label="Open report"
            />
          </Tooltip>
        )}
        {isTerminal(run.status) &&
          (confirmDelete ? (
            <Button size="sm" variant="danger" loading={busy} onClick={remove}>
              Confirm
            </Button>
          ) : (
            <Tooltip content="Delete this run and its stored events">
              <Button
                size="sm"
                variant="ghost"
                iconOnly
                icon={<Trash2 size={14} />}
                onClick={() => setConfirmDelete(true)}
                aria-label="Delete run"
              />
            </Tooltip>
          ))}
      </div>
    </li>
  )
})
