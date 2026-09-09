/**
 * Overview.
 *
 * The landing screen answers three questions in order: can I run anything yet,
 * what is this engine actually going to do, and what happened recently. Every
 * description of the engine is read from `/capabilities` so the screen cannot
 * drift from the backend.
 */

import { useMemo, type CSSProperties, type JSX, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  Check,
  Cpu,
  Database,
  FlaskConical,
  Inbox,
  KeyRound,
  Play,
  Route,
} from 'lucide-react'
import {
  Badge,
  Button,
  Card,
  Disclosure,
  EmptyState,
  InfoDot,
  Skeleton,
  Tooltip,
} from '@/components/ui/primitives'
import {
  duration,
  elapsedSince,
  humaniseId,
  number,
  percent,
  relativeTime,
  titleCase,
  tokens,
  usd,
} from '@/lib/format'
import { useAccount, useCapabilities, useNow, useParticipants, useRuns, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { Capabilities, Participant, PredictionType, RunStatus, RunSummary } from '@/lib/types'
import './overview.css'

/* The disclosure remembers a collapse rather than an expand, so the absent
   default reads as open on a first visit. */
const HOW_KEY = 'overview.how-it-works.collapsed'

export function OverviewPage(): JSX.Element {
  return (
    <div className="page">
      <ReadinessStrip />
      <HowItWorks />
      <div className="ov__split">
        <RecentActivity />
        <SampleData />
      </div>
    </div>
  )
}

/* --- Readiness ------------------------------------------------------------ */

interface ReadinessCheck {
  key: string
  title: string
  icon: ReactNode
  done: boolean
  loading: boolean
  detail: ReactNode
  summary: string
  action?: { label: string; onClick: () => void }
}

function ReadinessStrip() {
  const navigate = useNavigate()
  const openSettings = useApp((s) => s.openSettings)
  const startTour = useApp((s) => s.startTour)

  const account = useAccount()
  const participants = useParticipants()
  const settings = useSettings()

  const models = settings.data?.config.models
  const valid = (participants.data?.participants ?? []).filter((p) => p.valid)
  const roots = (participants.data?.roots ?? []).filter((r) => r.found > 0)

  const checks: ReadinessCheck[] = [
    {
      key: 'connection',
      title: 'Provider connected',
      icon: <KeyRound size={13} />,
      done: Boolean(account.data?.valid),
      loading: account.isLoading,
      summary: account.data?.valid
        ? account.data.remaining_usd != null
          ? `provider connected with ${usd(account.data.remaining_usd)} left`
          : 'provider connected'
        : 'no provider key',
      detail: account.isLoading ? (
        <Skeleton width={140} />
      ) : account.data?.valid ? (
        <span className="t-tiny muted">
          {account.data.label ? `${account.data.label}. ` : ''}
          {account.data.remaining_usd != null
            ? `${usd(account.data.remaining_usd)} of credit remaining`
            : 'Credit balance not reported by the provider'}
          {account.data.usage_usd != null ? `, ${usd(account.data.usage_usd)} used to date` : ''}
        </span>
      ) : (
        <span className="t-tiny muted">
          {account.data?.reason ?? 'No provider key is stored. The engine cannot call a model without one.'}
        </span>
      ),
      action: account.data?.valid ? undefined : { label: 'Add a key', onClick: () => openSettings('connection') },
    },
    {
      key: 'data',
      title: 'Data available',
      icon: <Database size={13} />,
      done: valid.length > 0,
      loading: participants.isLoading,
      summary: `${valid.length} participant${valid.length === 1 ? '' : 's'} found`,
      detail: participants.isLoading ? (
        <Skeleton width={160} />
      ) : valid.length > 0 ? (
        <span className="t-tiny muted">
          {valid.length} valid participant{valid.length === 1 ? '' : 's'} across{' '}
          {roots.length} root{roots.length === 1 ? '' : 's'}: {roots.map((r) => r.label).join(', ')}
        </span>
      ) : (
        <span className="t-tiny muted">
          No participant directory scanned clean. Point the workspace at a folder of participant inputs.
        </span>
      ),
      action: valid.length > 0 ? undefined : { label: 'Add a data root', onClick: () => openSettings('workspace') },
    },
    {
      key: 'model',
      title: 'Model chosen',
      icon: <Cpu size={13} />,
      done: Boolean(models?.default_model),
      loading: settings.isLoading,
      summary: models?.default_model ?? 'unset',
      detail: settings.isLoading ? (
        <Skeleton width={150} />
      ) : models?.default_model ? (
        <span className="t-tiny muted">
          <span className="mono">{models.default_model}</span>, reasoning effort{' '}
          {models.reasoning_effort === 'provider_default' ? 'left to the provider' : models.reasoning_effort}
        </span>
      ) : (
        <span className="t-tiny muted">No default model is set, so every role would fall back to the built in default.</span>
      ),
      action: models?.default_model ? undefined : { label: 'Choose a model', onClick: () => openSettings('models') },
    },
  ]

  const ready = checks.every((c) => c.done)

  if (ready) {
    return (
      <Card>
        <div className="ov__ready-line">
          <span
            className="ov__step-mark"
            style={{ background: 'var(--positive-soft)', color: 'var(--positive)', borderColor: 'transparent' }}
          >
            <Check size={14} />
          </span>
          <span className="grow t-small secondary">
            Ready to run: {checks[0].summary}, {checks[1].summary}, running{' '}
            <span className="mono">{checks[2].summary}</span>.
          </span>
          <Button variant="ghost" size="sm" onClick={startTour}>
            Take the tour
          </Button>
          <Button variant="primary" size="sm" icon={<Play size={13} />} onClick={() => navigate('/studio')}>
            Start a run
          </Button>
        </div>
      </Card>
    )
  }

  return (
    <Card
      title="Before the first run"
      subtitle="Three things have to be in place. Each one is a link to where it is set."
      actions={
        <Button variant="ghost" size="sm" onClick={startTour}>
          Take the tour
        </Button>
      }
    >
      <div className="ov__ready">
        {checks.map((check, index) => (
          <div key={check.key} className="ov__step" data-done={check.loading ? undefined : check.done}>
            <span className="ov__step-mark">{check.done ? <Check size={14} /> : check.icon}</span>
            <div className="stack gap-2 grow">
              <span className="t-small semibold">
                {index + 1}. {check.title}
              </span>
              {check.detail}
              {check.action && (
                <div>
                  <Button variant="primary" size="sm" onClick={check.action.onClick}>
                    {check.action.label}
                  </Button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

/* --- What the engine does ------------------------------------------------- */

function HowItWorks() {
  const collapsed = useApp((s) => Boolean(s.advancedOpen[HOW_KEY]))
  const toggleAdvanced = useApp((s) => s.toggleAdvanced)
  const { data, isLoading, error } = useCapabilities()

  return (
    <Disclosure
      title="How it works"
      subtitle={
        data
          ? `${data.stages.length} stages, ${data.agents.length} agents, ${data.tools.length} tools, ${data.prediction_types.length} task shapes`
          : 'The stages, agents, tools, and task shapes this build exposes'
      }
      open={!collapsed}
      onOpenChange={() => toggleAdvanced(HOW_KEY)}
      right={data ? <Badge mono>{`v${data.version}`}</Badge> : null}
    >
      {isLoading && <Skeleton height={160} />}
      {error && (
        <span className="t-small muted">
          The capability description could not be loaded, so this section is empty. The rest of the screen still works.
        </span>
      )}
      {data && (
        <div className="stack gap-6">
          <StageDiagram stages={data.stages} />
          <AgentRoster agents={data.agents} stages={data.stages} />
          <ToolRoster tools={data.tools} />
          <PredictionTypes types={data.prediction_types} />
        </div>
      )}
    </Disclosure>
  )
}

function StageDiagram({ stages }: { stages: string[] }) {
  if (stages.length === 0) return null
  return (
    <section className="stack gap-3">
      <span className="eyebrow">Stages</span>
      <div className="ov__stages">
        {stages.map((stage, index) => (
          <div key={stage} className="ov__stage">
            <span className="ov__stage-index">{index + 1}</span>
            <span className="t-small semibold">{stage}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function AgentRoster({ agents, stages }: { agents: Capabilities['agents']; stages: string[] }) {
  if (agents.length === 0) return null
  return (
    <section className="stack gap-3">
      <span className="eyebrow">Agents</span>
      <div className="grid grid--3">
        {agents.map((agent) => (
          <div key={agent.role} className="ov__agent">
            <div className="row gap-2">
              <span className="t-small semibold grow truncate">{agent.label}</span>
              <Tooltip content={stages[agent.stage] ?? ''}>
                <Badge tone="accent">Stage {agent.stage}</Badge>
              </Tooltip>
            </div>
            <span className="t-tiny muted">{agent.summary}</span>
            <span className="t-micro faint mono">{agent.role}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function ToolRoster({ tools }: { tools: Capabilities['tools'] }) {
  const families = useMemo(() => {
    const grouped = new Map<string, Capabilities['tools']>()
    for (const tool of tools) {
      const key = tool.family || 'other'
      const bucket = grouped.get(key)
      if (bucket) bucket.push(tool)
      else grouped.set(key, [tool])
    }
    return [...grouped.entries()]
  }, [tools])

  if (families.length === 0) return null

  return (
    <section className="stack gap-3">
      <span className="eyebrow">Tools</span>
      <div className="ov__families">
        {families.map(([family, rows], index) => (
          <div key={family} className="stack gap-2" style={{ '--fam': `var(--fam-${index % 8})` } as CSSProperties}>
            <div className="row gap-2">
              <span className="ov__family-dot" style={{ '--fam': `var(--fam-${index % 8})` } as CSSProperties} />
              <span className="t-tiny semibold" style={{ color: 'var(--fam)' }}>
                {titleCase(family)}
              </span>
              <span className="t-micro faint">{rows.length}</span>
            </div>
            <div className="ov__tools">
              {rows.map((tool) => (
                <div key={tool.name} className="ov__tool">
                  <span className="t-tiny semibold">{tool.name}</span>
                  {tool.summary && <span className="t-micro muted">{tool.summary}</span>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function PredictionTypes({ types }: { types: Capabilities['prediction_types'] }) {
  const navigate = useNavigate()
  const setTask = useApp((s) => s.setTask)
  if (types.length === 0) return null

  const use = (value: PredictionType) => {
    setTask({ prediction_type: value })
    navigate('/studio')
  }

  return (
    <section className="stack gap-3">
      <span className="eyebrow">Task shapes</span>
      <div className="grid grid--3">
        {types.map((type) => (
          <div key={type.value} className="ov__ptype">
            <span className="t-small semibold">{type.label}</span>
            <span className="t-tiny muted grow">{type.summary}</span>
            <div className="row gap-1 wrap">
              {type.needs.map((need) => (
                <Badge key={need} outline>
                  {titleCase(need)}
                </Badge>
              ))}
            </div>
            <div>
              <Button size="sm" icon={<ArrowRight size={13} />} onClick={() => use(type.value)}>
                Use this
              </Button>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
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

function RecentActivity() {
  const navigate = useNavigate()
  const { data, isLoading } = useRuns(true)
  const runs = data?.runs ?? []
  const live = runs.some((r) => r.status === 'running' || r.status === 'queued')
  useNow(live)

  const stats = useMemo(() => {
    const today = new Date().toDateString()
    const runsToday = runs.filter((r) => {
      const parsed = Date.parse(r.created_at)
      return !Number.isNaN(parsed) && new Date(parsed).toDateString() === today
    }).length
    const spend = runs.reduce((sum, r) => sum + (r.cost.usd ?? 0), 0)
    const durations = runs.map(runSeconds).filter((s): s is number => s !== null && s > 0)
    const terminal = runs.filter((r) => r.status === 'succeeded' || r.status === 'failed')
    return {
      runsToday,
      spend,
      average: durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null,
      success: terminal.length ? terminal.filter((r) => r.status === 'succeeded').length / terminal.length : null,
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
        <Tile label="Total spend" value={usd(stats.spend)} />
        <Tile label="Average duration" value={duration(stats.average)} />
        <Tile
          label="Success rate"
          value={stats.success === null ? '-' : percent(stats.success, 0)}
          meta={stats.success === null ? 'no finished runs yet' : undefined}
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
                <tr key={run.id} className="ov__run-row" onClick={() => navigate(`/runs/${run.id}`)}>
                  <td>
                    <span className="stack" style={{ gap: 0 }}>
                      <span className="semibold truncate">{run.participant_id || humaniseId(run.id)}</span>
                      <span className="t-micro faint truncate">{run.task.display_name || run.label}</span>
                    </span>
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

/* --- Bundled sample data --------------------------------------------------- */

function SampleData() {
  const navigate = useNavigate()
  const setSelected = useApp((s) => s.setSelected)
  const { data, isLoading } = useParticipants()

  const bundled = useMemo(() => {
    const roots = (data?.roots ?? []).filter((r) => r.bundled)
    const participants = (data?.participants ?? []).filter((p) =>
      roots.some((r) => p.directory.startsWith(r.root)),
    )
    return { roots, participants }
  }, [data])

  const configure = (participant: Participant) => {
    setSelected([participant])
    navigate('/studio')
  }

  return (
    <Card title="Bundled sample data" subtitle="Exercise the whole flow before connecting real inputs">
      <div className="stack gap-4">
        <span className="t-small secondary">
          A small set of synthetic participants ships with the engine. They carry the same file layout and ontology
          coverage as real inputs, so a run over them goes through every stage and produces a real report. Nothing here
          is patient data.
        </span>

        {isLoading && <Skeleton height={80} />}

        {!isLoading && bundled.participants.length === 0 && (
          <EmptyState
            icon={<FlaskConical size={20} />}
            title="No bundled participants found"
            body="The sample directory that ships with the engine was not picked up by the scan."
          />
        )}

        {bundled.participants.length > 0 && (
          <div className="stack">
            {bundled.participants.map((participant) => (
              <div key={participant.directory} className="ov__sample">
                <span className="stack grow" style={{ gap: 0, minWidth: 0 }}>
                  <span className="t-small semibold truncate">{participant.id}</span>
                  <span className="t-micro faint">
                    {tokens(participant.input_tokens)} input tokens, {participant.domains.length} domain
                    {participant.domains.length === 1 ? '' : 's'}
                    {participant.valid ? '' : ', incomplete'}
                  </span>
                </span>
                <Button size="sm" variant="ghost" icon={<Route size={13} />} onClick={() => configure(participant)}>
                  Configure a run with this one
                </Button>
              </div>
            ))}
          </div>
        )}

        {bundled.roots.length > 0 && (
          <div className="row gap-2">
            <span className="t-micro faint truncate mono grow">{bundled.roots.map((r) => r.root).join(' ')}</span>
            <InfoDot label="Where the samples live">
              The scanner walks every configured data root. The bundled set sits inside the package itself, which is why
              it is present before any workspace folder is added. Add your own root in Settings to have real
              participants appear alongside these.
            </InfoDot>
          </div>
        )}
      </div>
    </Card>
  )
}
