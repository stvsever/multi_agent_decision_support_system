/**
 * The live run console.
 *
 * Left: the execution graph as it fills in. Right: the timeline, the raw event
 * feed, and the results as each agent produces them. Everything is driven by
 * one server-sent event stream, with a polling fallback behind it.
 */

import clsx from 'clsx'
import { useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, CircleSlash, Download, FileText, Layers, RefreshCw, X } from 'lucide-react'
import { memo, useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Badge, Button, Callout, EmptyState, Skeleton, Tabs, Tooltip } from '@/components/ui/primitives'
import { api } from '@/lib/api'
import { duration as formatDuration, elapsedSince, tokens as formatTokens, usd } from '@/lib/format'
import { queryKeys, useNow, useRunStream, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { GraphNode, RunDetail, RunState, RunStatus } from '@/lib/types'
import { AuditView } from './AuditView'
import { CostPanel } from './CostPanel'
import { CriticPanel } from './CriticPanel'
import { EventsPanel } from './EventsPanel'
import { FailurePanel } from './FailurePanel'
import { FlowCanvas } from './FlowCanvas'
import { LogsPanel } from './LogsPanel'
import { NodeInspector } from './NodeInspector'
import { PredictionPanel } from './PredictionPanel'
import { StageRail } from './StageRail'
import { TimelinePanel } from './TimelinePanel'
import { STAGE_NAMES, isActive, statusTone, taskLine, taskModeLabel, taskSpecFromSummary, verdictTone } from './runUtils'
import './run.css'

type TabKey = 'timeline' | 'events' | 'prediction' | 'critic' | 'cost' | 'logs'

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  cancelling: 'Cancelling',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

const EMPTY_STATE: RunState = {
  status: 'Waiting',
  current_stage: -1,
  stages: STAGE_NAMES,
  steps: [],
  history: [],
  iteration: 1,
  total_tokens: 0,
  progress: 0,
  max_steps: 1,
  completed: false,
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const { detail, connected } = useRunStream(runId ?? null)
  const setActiveRunId = useApp((s) => s.setActiveRunId)
  const [tab, setTab] = useState<TabKey>('timeline')
  const [selected, setSelected] = useState<GraphNode | null>(null)

  useEffect(() => {
    setActiveRunId(runId ?? null)
    return () => setActiveRunId(null)
  }, [runId, setActiveRunId])

  const onSelectNode = useCallback((node: GraphNode | null) => setSelected(node), [])
  const closeInspector = useCallback(() => setSelected(null), [])

  if (!runId) {
    return (
      <div className="page">
        <EmptyState
          title="No run selected"
          body="Pick a run from the history to watch or review it."
          action={
            <Link to="/runs">
              <Button variant="primary">Back to runs</Button>
            </Link>
          }
        />
      </div>
    )
  }

  if (!detail) {
    return (
      <div className="page">
        <Skeleton height={96} radius={14} />
        <Skeleton height={72} radius={14} />
        <div className="run-body">
          <Skeleton height={440} radius={14} />
          <Skeleton height={440} radius={14} />
        </div>
      </div>
    )
  }

  // An audit calls no model and publishes no plan, so the live console would be
  // an empty frame. It gets a screen that shows what an audit actually proves.
  if (detail.audit) return <AuditView detail={detail} />

  return (
    <RunConsole
      detail={detail}
      connected={connected}
      tab={tab}
      onTab={setTab}
      selected={selected}
      onSelectNode={onSelectNode}
      onCloseInspector={closeInspector}
    />
  )
}

function RunConsole({
  detail,
  connected,
  tab,
  onTab,
  selected,
  onSelectNode,
  onCloseInspector,
}: {
  detail: RunDetail
  connected: boolean
  tab: TabKey
  onTab: (next: TabKey) => void
  selected: GraphNode | null
  onSelectNode: (node: GraphNode | null) => void
  onCloseInspector: () => void
}) {
  const state = detail.state ?? EMPTY_STATE
  const stages = state.stages?.length ? state.stages : STAGE_NAMES
  const steps = state.steps ?? []
  const history = state.history ?? []
  const events = detail.events ?? []
  const logs = detail.logs ?? []
  const running = isActive(detail.status)

  const appearance = useSettings().data?.config.appearance

  const stepCount = steps.length + history.length
  const tabs = useMemo(
    () =>
      [
        { value: 'timeline' as const, label: 'Timeline', badge: <span className="run-tabcount tabular">{stepCount}</span> },
        { value: 'events' as const, label: 'Events', badge: <span className="run-tabcount tabular">{events.length}</span> },
        {
          value: 'prediction' as const,
          label: 'Prediction',
          badge: state.prediction ? <span className="run-tabdot" aria-hidden /> : undefined,
        },
        {
          value: 'critic' as const,
          label: 'Critic',
          badge: state.critic ? <span className="run-tabdot" aria-hidden /> : undefined,
        },
        { value: 'cost' as const, label: 'Cost' },
        { value: 'logs' as const, label: 'Logs', badge: <span className="run-tabcount tabular">{logs.length}</span> },
      ],
    [stepCount, events.length, logs.length, state.prediction, state.critic],
  )

  return (
    <div className="page page--wide run-page">
      <RunHeader detail={detail} connected={connected} running={running} />

      {detail.status === 'failed' && <FailurePanel detail={detail} />}

      {detail.status === 'cancelled' && (
        <Callout tone="neutral" icon={<CircleSlash size={15} />} title="This run was cancelled">
          Everything below is what the engine had produced when the worker was stopped.
        </Callout>
      )}

      <StageRail
        stages={stages}
        currentStage={state.current_stage ?? -1}
        events={events}
        running={running}
        failed={detail.status === 'failed'}
        iteration={state.iteration ?? 1}
        maxIterations={state.max_iterations}
      />

      <div className="run-body">
        <section className="run-canvas card">
          <header className="card__header">
            <div className="stack" style={{ gap: 1, minWidth: 0 }}>
              <span className="card__title">Execution graph</span>
              <span className="t-tiny muted truncate">
                {detail.graph
                  ? `${detail.graph.meta.step_count} steps, depth ${detail.graph.meta.depth}, up to ${detail.graph.meta.max_parallel} in parallel`
                  : 'The orchestrator has not published a plan yet'}
              </span>
            </div>
          </header>
          <div className="run-canvas__stage">
            <FlowCanvas
              graph={detail.graph}
              steps={steps}
              currentStage={state.current_stage ?? -1}
              iteration={state.iteration ?? 1}
              running={running}
              animate={appearance?.flow_animate_edges ?? true}
              direction={appearance?.flow_direction ?? 'LR'}
              edgeStyle={appearance?.flow_edge_style ?? 'bezier'}
              showTokens={appearance?.flow_show_tokens ?? true}
              onSelectNode={onSelectNode}
              className="run-canvas__flow"
            />
          </div>
          <NodeInspector node={selected} steps={steps} onClose={onCloseInspector} />
        </section>

        <section className="run-side card">
          <header className="card__header run-side__head">
            <Tabs<TabKey> value={tab} options={tabs} onChange={onTab} spread />
          </header>
          <div className="run-side__body">
            {tab === 'timeline' && <TimelinePanel steps={steps} history={history} graph={detail.graph} />}
            {tab === 'events' && <EventsPanel events={events} />}
            {tab === 'prediction' && <PredictionPanel prediction={state.prediction} />}
            {tab === 'critic' && <CriticPanel critic={state.critic} />}
            {tab === 'cost' && <CostPanel cost={detail.cost} estimate={detail.estimate} />}
            {tab === 'logs' && <LogsPanel logs={logs} />}
          </div>
        </section>
      </div>
    </div>
  )
}

/* --- Header --------------------------------------------------------------- */

function RunHeader({ detail, connected, running }: { detail: RunDetail; connected: boolean; running: boolean }) {
  const client = useQueryClient()
  const navigate = useNavigate()
  const notify = useApp((s) => s.notify)
  const setTask = useApp((s) => s.setTask)
  const [busy, setBusy] = useState(false)

  const state = detail.state ?? EMPTY_STATE
  const spent = detail.cost?.usd ?? null
  const projected = detail.estimate?.usd ?? null
  const share = spent !== null && projected ? Math.max(0, Math.min(1.35, spent / projected)) : 0

  const cancel = useCallback(async () => {
    setBusy(true)
    try {
      await api.runs.cancel(detail.id)
      notify({ tone: 'info', title: 'Cancelling run', body: detail.participant_id })
    } catch (error) {
      notify({ tone: 'critical', title: 'Could not cancel', body: (error as Error).message })
    } finally {
      setBusy(false)
      client.invalidateQueries({ queryKey: queryKeys.runs })
    }
  }, [client, detail.id, detail.participant_id, notify])

  const rerun = useCallback(() => {
    if (detail.task) setTask(taskSpecFromSummary(detail.task))
    notify({
      tone: 'info',
      title: 'Configuration carried over',
      body: 'Select the participant to launch the same task again.',
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
        {detail.batch_id && (
          <Link to="/batch" state={{ batchId: detail.batch_id }} className="run-back">
            <Layers size={13} />
            <span className="t-tiny">Part of a batch</span>
          </Link>
        )}
        <div className="grow" />
        <ConnectionDot connected={connected} running={running} />
        <div className="row gap-2">
          {running && (
            <Button size="sm" variant="secondary" icon={<X size={14} />} loading={busy} onClick={cancel}>
              Cancel
            </Button>
          )}
          {detail.status === 'succeeded' && (
            <>
              <Tooltip content="Download the service-composed summary of this run as a PDF">
                <a
                  href={api.reports.pdfUrl({ run_id: detail.id })}
                  target="_blank"
                  rel="noreferrer"
                  className="run-plainlink"
                >
                  <Button size="sm" variant="secondary" icon={<Download size={14} />}>
                    PDF
                  </Button>
                </a>
              </Tooltip>
              <Button
                size="sm"
                variant="secondary"
                icon={<FileText size={14} />}
                onClick={() =>
                  navigate(`/reports/${encodeURIComponent(detail.participant_id)}?run_id=${detail.id}`)
                }
              >
                Report
              </Button>
            </>
          )}
          <Tooltip content="Open the studio with this task prefilled">
            <Button size="sm" variant="ghost" icon={<RefreshCw size={14} />} onClick={rerun}>
              Re-run
            </Button>
          </Tooltip>
        </div>
      </div>

      <div className="run-head__main">
        <div className="stack gap-2" style={{ minWidth: 0 }}>
          <div className="row gap-3 wrap">
            <h1 className="page__title truncate">{detail.participant_id}</h1>
            <Badge tone={statusTone(detail.status)}>{STATUS_LABEL[detail.status]}</Badge>
            <Badge tone="neutral">{taskModeLabel(detail.task)}</Badge>
            {detail.verdict && <Badge tone={verdictTone(detail.verdict)}>{detail.verdict}</Badge>}
          </div>
          <span className="t-small muted truncate">
            {detail.label !== detail.participant_id ? `${detail.label} - ` : ''}
            {taskLine(detail.task)}
          </span>
          <span className="t-tiny secondary truncate">{state.status}</span>
        </div>

        <div className="run-head__stats">
          <div className="stat">
            <span className="stat__label">{detail.finished_at ? 'Duration' : 'Elapsed'}</span>
            <span className="stat__value">
              <LiveElapsed startedAt={detail.started_at} finishedAt={detail.finished_at} running={running} />
            </span>
            <span className="stat__meta tabular">
              {state.progress ?? 0} of {state.max_steps ?? 0} steps
            </span>
          </div>
          <div className="stat">
            <span className="stat__label">Tokens</span>
            <span className="stat__value tabular">{formatTokens(detail.cost?.total_tokens || state.total_tokens || 0)}</span>
            <span className="stat__meta tabular">
              {detail.estimate?.total_tokens ? `of ~${formatTokens(detail.estimate.total_tokens)} projected` : 'no projection'}
            </span>
          </div>
          <div className="stat run-head__cost">
            <span className="stat__label">Cost</span>
            <span className="stat__value tabular">{usd(spent)}</span>
            <span className="stat__meta tabular">
              {projected === null ? 'no projection' : `of ~${usd(projected)} projected`}
            </span>
            <span className="run-costbar" aria-hidden>
              <span
                className={clsx('run-costbar__fill', share > 1 && 'run-costbar__fill--over')}
                style={{ width: `${Math.round(Math.min(1, share) * 100)}%` }}
              />
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}

const LiveElapsed = memo(function LiveElapsed({
  startedAt,
  finishedAt,
  running,
}: {
  startedAt: string | null
  finishedAt: string | null
  running: boolean
}) {
  const now = useNow(running)
  if (!startedAt) return <span className="tabular">-</span>
  const seconds = finishedAt
    ? elapsedSince(startedAt, finishedAt)
    : Math.max(0, (now - Date.parse(startedAt)) / 1000)
  return <span className="tabular">{formatDuration(seconds)}</span>
})

function ConnectionDot({ connected, running }: { connected: boolean; running: boolean }) {
  const label = connected && running ? 'Live stream' : running ? 'Polling' : 'Stream closed'
  const hint =
    connected && running
      ? 'Attached to the server-sent event stream. Updates arrive as the engine emits them.'
      : running
        ? 'The event stream is not attached, so the screen refreshes by polling every two seconds.'
        : 'This run has finished, so there is nothing left to stream.'
  return (
    <Tooltip content={hint}>
      <span className={clsx('run-conn', connected && running && 'run-conn--live', !connected && running && 'run-conn--poll')}>
        <span className="run-conn__dot" aria-hidden />
        <span className="t-tiny">{label}</span>
      </span>
    </Tooltip>
  )
}
