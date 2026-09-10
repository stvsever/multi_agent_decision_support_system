/**
 * Overview.
 *
 * The landing screen explains the method and reports what has run lately. It
 * states nothing about readiness while everything is in order: only a genuine
 * blocker earns a line here, and it comes with the action that clears it.
 */

import { useMemo, type JSX } from 'react'
import { useNavigate } from 'react-router-dom'
import { Database, Inbox, KeyRound, Play } from 'lucide-react'
import { Badge, Button, Callout, Card, EmptyState, Skeleton } from '@/components/ui/primitives'
import { duration, elapsedSince, humaniseId, number, percent, relativeTime, titleCase, usd } from '@/lib/format'
import { useAccount, useNow, useParticipants, useRuns, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import { MethodologyFilm } from './MethodologyFilm'
import type { RunStatus, RunSummary } from '@/lib/types'
import './overview.css'

export function OverviewPage(): JSX.Element {
  const navigate = useNavigate()
  const startTour = useApp((state) => state.startTour)

  return (
    <div className="page">
      <header className="page__head">
        <div className="stack gap-2">
          <span className="page__title">Overview</span>
          <p className="page__lede">
            How the engine turns one participant's measurements into an evidence-backed answer, and what has run
            lately.
          </p>
        </div>
        <div className="row gap-2">
          <Button variant="ghost" onClick={startTour}>
            Take the tour
          </Button>
          <Button variant="primary" icon={<Play size={13} />} onClick={() => navigate('/studio')}>
            Start a run
          </Button>
        </div>
      </header>

      <Blocker />
      <MethodologyFilm />
      <RecentActivity />
    </div>
  )
}

/* --- The one thing in the way --------------------------------------------- */

/**
 * At most one notice, and only for a state that stops a run outright. A healthy
 * workspace renders nothing at all.
 */
function Blocker() {
  const navigate = useNavigate()
  const openSettings = useApp((state) => state.openSettings)
  const settings = useSettings()
  const account = useAccount()
  const participants = useParticipants()

  const backend = settings.data?.config.connection.backend
  const keyMissing = backend !== 'local' && account.isSuccess && !account.data.valid
  const validCount = (participants.data?.participants ?? []).filter((row) => row.valid).length

  if (keyMissing) {
    return (
      <Callout
        tone="caution"
        icon={<KeyRound size={15} />}
        title="No provider key is stored"
        action={
          <Button size="sm" variant="primary" onClick={() => openSettings('connection')}>
            Add a key
          </Button>
        }
      >
        {account.data.reason ?? 'The engine cannot call a model until a key is saved.'}
      </Callout>
    )
  }

  if (participants.isSuccess && validCount === 0) {
    return (
      <Callout
        tone="caution"
        icon={<Database size={15} />}
        title="No valid participant inputs were found"
        action={
          <Button size="sm" variant="primary" onClick={() => navigate('/studio')}>
            Go to Configure
          </Button>
        }
      >
        Every configured data root scanned clean of usable participants, so there is nothing to run against yet.
      </Callout>
    )
  }

  return null
}

/* --- Recent activity ------------------------------------------------------ */

const STATUS_TONE: Record<RunStatus, 'neutral' | 'accent' | 'positive' | 'caution' | 'critical' | 'info'> = {
  queued: 'neutral',
  running: 'accent',
  cancelling: 'caution',
  succeeded: 'positive',
  failed: 'critical',
  cancelled: 'neutral',
}

function runSeconds(run: RunSummary): number | null {
  return elapsedSince(run.started_at, run.finished_at)
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

function RecentActivity() {
  const navigate = useNavigate()
  const { data, isLoading } = useRuns(true)
  const runs = useMemo(() => data?.runs ?? [], [data])
  const live = runs.some((run) => run.status === 'running' || run.status === 'queued')
  useNow(live)

  const stats = useMemo(() => {
    const today = new Date().toDateString()
    const runsToday = runs.filter((run) => {
      const parsed = Date.parse(run.created_at)
      return !Number.isNaN(parsed) && new Date(parsed).toDateString() === today
    }).length
    const durations = runs.map(runSeconds).filter((value): value is number => value !== null && value > 0)
    const terminal = runs.filter((run) => run.status === 'succeeded' || run.status === 'failed')
    const costs = runs
      .filter((run) => run.status === 'succeeded')
      .map((run) => run.cost.usd)
      .filter((value): value is number => value !== null && value !== undefined)
    return {
      runsToday,
      average: durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null,
      success: terminal.length ? terminal.filter((run) => run.status === 'succeeded').length / terminal.length : null,
      medianCost: median(costs),
      costCount: costs.length,
    }
  }, [runs])

  return (
    <Card
      title="Recent activity"
      subtitle={runs.length > 0 ? `${runs.length} run${runs.length === 1 ? '' : 's'} recorded` : undefined}
      actions={
        runs.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={() => navigate('/runs')}>
            All runs
          </Button>
        ) : undefined
      }
      flush
    >
      <div className="ov__tiles">
        <Tile label="Runs today" value={number(stats.runsToday)} />
        <Tile label="Average duration" value={duration(stats.average)} />
        <Tile
          label="Success rate"
          value={stats.success === null ? '-' : percent(stats.success, 0)}
          meta={stats.success === null ? 'no finished runs yet' : undefined}
        />
        <Tile
          label="Median cost per run"
          value={stats.medianCost === null ? '-' : usd(stats.medianCost)}
          meta={stats.medianCost === null ? 'no completed run to measure' : `across ${stats.costCount} completed`}
        />
      </div>

      {isLoading && (
        <div className="stack gap-2" style={{ padding: 'var(--s-5)' }}>
          <Skeleton height={18} />
          <Skeleton height={18} />
          <Skeleton height={18} />
        </div>
      )}

      {!isLoading && runs.length === 0 && (
        <EmptyState
          icon={<Inbox size={20} />}
          title="No runs yet"
          body="Once a run finishes it appears here with its verdict, duration, and cost."
          action={
            <Button variant="primary" size="sm" icon={<Play size={13} />} onClick={() => navigate('/studio')}>
              Configure a run
            </Button>
          }
        />
      )}

      {runs.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Participant</th>
                <th>Status</th>
                <th>Verdict</th>
                <th className="num">Duration</th>
                <th className="num">Cost</th>
                <th className="num">Started</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 8).map((run) => (
                <tr key={run.id}>
                  <td>
                    {/* A real control, so the row is reachable by keyboard. */}
                    <button type="button" className="ov__row-link" onClick={() => navigate(`/runs/${run.id}`)}>
                      <span className="semibold truncate">{run.participant_id || humaniseId(run.id)}</span>
                      <span className="t-micro faint truncate">{run.task.display_name || run.label}</span>
                    </button>
                  </td>
                  <td>
                    <Badge tone={STATUS_TONE[run.status]}>{run.status}</Badge>
                  </td>
                  <td>
                    {run.verdict ? (
                      <Badge tone={run.verdict === 'SATISFACTORY' ? 'positive' : 'caution'}>
                        {titleCase(run.verdict)}
                      </Badge>
                    ) : (
                      <span className="faint">-</span>
                    )}
                  </td>
                  <td className="num">{duration(runSeconds(run))}</td>
                  <td className="num">{run.cost.usd == null ? '-' : usd(run.cost.usd)}</td>
                  <td className="num muted">{relativeTime(run.started_at ?? run.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

function Tile({ label, value, meta }: { label: string; value: string; meta?: string }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      {meta && <span className="stat__meta">{meta}</span>}
    </div>
  )
}
