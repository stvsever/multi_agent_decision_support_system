/**
 * Batch monitor.
 *
 * One list of batches, one grid of the selected batch's runs, and a stacked
 * bar that makes the queued/running/finished split readable at a glance while
 * the batch is still moving.
 */

import { useMemo } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Layers, Square } from 'lucide-react'
import { Badge, Button, Card, EmptyState, Progress, Skeleton } from '@/components/ui/primitives'
import { api } from '@/lib/api'
import { duration, elapsedSince, relativeTime, titleCase, tokens, usd } from '@/lib/format'
import { queryKeys, useBatches, useNow } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { BatchStatus, RunSummary } from '@/lib/types'
import { BATCH_TONE, RUN_TONE } from './util'

const SEGMENTS: { key: string; label: string; colour: string; match: (run: RunSummary) => boolean }[] = [
  { key: 'queued', label: 'Queued', colour: 'var(--text-faint)', match: (r) => r.status === 'queued' },
  {
    key: 'running',
    label: 'Running',
    colour: 'var(--accent)',
    match: (r) => r.status === 'running' || r.status === 'cancelling',
  },
  { key: 'succeeded', label: 'Succeeded', colour: 'var(--positive)', match: (r) => r.status === 'succeeded' },
  { key: 'failed', label: 'Failed', colour: 'var(--critical)', match: (r) => r.status === 'failed' },
  { key: 'cancelled', label: 'Cancelled', colour: 'var(--line-strong)', match: (r) => r.status === 'cancelled' },
]

/** Cancel a batch and refresh the two lists that show it. */
function useCancelBatch() {
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  return async (batchId: string) => {
    try {
      await api.batches.cancel(batchId)
      client.invalidateQueries({ queryKey: queryKeys.batches })
      client.invalidateQueries({ queryKey: queryKeys.runs })
      notify({ tone: 'info', title: 'Cancelling', body: 'Running participants stop at their next checkpoint.' })
    } catch (error) {
      notify({ tone: 'critical', title: 'Could not cancel', body: error instanceof Error ? error.message : String(error) })
    }
  }
}

export function BatchMonitor({
  activeId,
  onSelect,
}: {
  activeId: string | null
  onSelect: (batchId: string) => void
}) {
  const { data, isLoading } = useBatches(true)
  const batches = data?.batches ?? []
  const active = batches.find((b) => b.id === activeId) ?? batches[0] ?? null

  useNow(batches.some((b) => b.status === 'running'))

  if (isLoading) {
    return (
      <div className="stack gap-3">
        <Skeleton height={64} />
        <Skeleton height={64} />
      </div>
    )
  }

  if (batches.length === 0) {
    return (
      <Card>
        <EmptyState
          icon={<Layers size={20} />}
          title="No batches yet"
          body="Compose a cohort and launch it. Every batch stays here with its runs, spend, and verdicts."
        />
      </Card>
    )
  }

  return (
    <div className="bt__monitor">
      <Card title="Batches" subtitle={`${batches.length} recorded`} flush>
        <div className="bt__list">
          {batches.map((batch) => (
            <BatchRow
              key={batch.id}
              batch={batch}
              current={batch.id === active?.id}
              onSelect={() => onSelect(batch.id)}
            />
          ))}
        </div>
      </Card>

      {active && <BatchDetail batch={active} />}
    </div>
  )
}

function BatchRow({
  batch,
  current,
  onSelect,
}: {
  batch: BatchStatus
  current: boolean
  onSelect: () => void
}) {
  const cancel = useCancelBatch()
  return (
    <div
      className="bt__batch"
      role="button"
      tabIndex={0}
      aria-current={current}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect()
        }
      }}
    >
      <div className="row gap-2">
        <span className="t-small semibold grow truncate">{batch.label}</span>
        {batch.status === 'running' ? (
          <Button
            size="sm"
            variant="ghost"
            icon={<Square size={12} />}
            onClick={(event) => {
              event.stopPropagation()
              void cancel(batch.id)
            }}
          >
            Cancel
          </Button>
        ) : (
          <Badge tone={BATCH_TONE[batch.status] ?? 'neutral'}>{batch.status}</Badge>
        )}
      </div>
      <Progress value={batch.completed} max={Math.max(1, batch.total)} />
      <div className="row gap-3 t-micro muted wrap">
        <span className="tabular">
          {batch.completed} of {batch.total}
        </span>
        <span className="tabular" style={{ color: 'var(--positive)' }}>
          {batch.succeeded} ok
        </span>
        <span className="tabular" style={{ color: batch.failed > 0 ? 'var(--critical)' : undefined }}>
          {batch.failed} failed
        </span>
        <span className="tabular">x{batch.concurrency} parallel</span>
        <span className="tabular grow" style={{ textAlign: 'right' }}>
          {usd(batch.spend_usd)}
        </span>
      </div>
      <span className="t-micro faint">{relativeTime(batch.created_at)}</span>
    </div>
  )
}

function BatchDetail({ batch }: { batch: BatchStatus }) {
  const navigate = useNavigate()
  const cancel = useCancelBatch()

  const counts = useMemo(
    () => SEGMENTS.map((segment) => ({ ...segment, count: batch.runs.filter(segment.match).length })),
    [batch.runs],
  )
  const total = Math.max(1, batch.runs.length)
  const running = batch.status === 'running'

  const projection = batch.runs.reduce((sum, run) => sum + (run.estimate.usd ?? 0), 0)
  const verdicts = useMemo(() => {
    const table = new Map<string, number>()
    for (const run of batch.runs) {
      if (!run.verdict) continue
      table.set(run.verdict, (table.get(run.verdict) ?? 0) + 1)
    }
    return [...table.entries()]
  }, [batch.runs])

  return (
    <Card
      title={batch.label}
      subtitle={`${batch.id}, started ${relativeTime(batch.created_at)}`}
      actions={
        running ? (
          <Button size="sm" variant="danger" icon={<Square size={13} />} onClick={() => void cancel(batch.id)}>
            Cancel batch
          </Button>
        ) : (
          <Badge tone={BATCH_TONE[batch.status] ?? 'neutral'}>{batch.status}</Badge>
        )
      }
    >
      <div className="stack gap-5">
        <div className="stack gap-2">
          <div className="bt__stack">
            {counts.map((segment) => (
              <span
                key={segment.key}
                className="bt__stack-seg"
                style={{ width: `${(segment.count / total) * 100}%`, background: segment.colour }}
              />
            ))}
          </div>
          <div className="bt__legend">
            {counts.map((segment) => (
              <span key={segment.key} className="row gap-2 t-micro muted">
                <span className="bt__legend-dot" style={{ background: segment.colour }} />
                {segment.label} {segment.count}
              </span>
            ))}
          </div>
        </div>

        {!running && (
          <div className="bt__summary">
            <div className="stat">
              <span className="stat__label">Total duration</span>
              <span className="stat__value">{duration(elapsedSince(batch.created_at, batch.finished_at))}</span>
            </div>
            <div className="stat">
              <span className="stat__label">Spend</span>
              <span className="stat__value">{usd(batch.spend_usd)}</span>
              <span className="stat__meta">
                {projection > 0 ? `projected ${usd(projection)}` : 'no projection recorded'}
              </span>
            </div>
            <div className="stat">
              <span className="stat__label">Verdicts</span>
              {verdicts.length === 0 ? (
                <span className="stat__value">-</span>
              ) : (
                <span className="row gap-2 wrap" style={{ marginTop: 4 }}>
                  {verdicts.map(([verdict, count]) => (
                    <Badge key={verdict} tone={verdict === 'SATISFACTORY' ? 'positive' : 'caution'}>
                      {titleCase(verdict)} {count}
                    </Badge>
                  ))}
                </span>
              )}
            </div>
          </div>
        )}

        {batch.runs.length === 0 ? (
          <span className="t-small muted">This batch has no runs attached.</span>
        ) : (
          <div className="bt__tiles">
            {batch.runs.map((run) => (
              <RunTile key={run.id} run={run} onOpen={() => navigate(`/runs/${run.id}`)} />
            ))}
          </div>
        )}
      </div>
    </Card>
  )
}

function RunTile({ run, onOpen }: { run: RunSummary; onOpen: () => void }) {
  const finished = run.status === 'succeeded' || run.status === 'failed' || run.status === 'cancelled'
  const fraction = finished ? 1 : Math.min(1, run.progress / Math.max(1, run.max_steps))

  return (
    <button type="button" className="bt__tile" data-status={run.status} onClick={onOpen}>
      <div className="row gap-2">
        <span className="t-small semibold grow truncate">{run.participant_id}</span>
        <Badge tone={RUN_TONE[run.status]}>{run.status}</Badge>
      </div>
      <Progress value={fraction} />
      <div className="bt__tile-foot t-micro muted">
        <span>{tokens(run.cost.total_tokens)} tok</span>
        <span>{run.cost.usd == null ? '-' : usd(run.cost.usd)}</span>
      </div>
      {run.verdict ? (
        <Badge tone={run.verdict === 'SATISFACTORY' ? 'positive' : 'caution'}>{titleCase(run.verdict)}</Badge>
      ) : run.error ? (
        <span className="t-micro truncate" style={{ color: 'var(--critical)' }}>
          {run.error}
        </span>
      ) : (
        <span className="t-micro faint">{duration(elapsedSince(run.started_at, run.finished_at))}</span>
      )}
    </button>
  )
}
