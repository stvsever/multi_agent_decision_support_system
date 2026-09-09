/**
 * Batch composer.
 *
 * Picks the cohort, shows the task that was designed in Configure, sets the
 * two concurrency-shaped options, and projects the cost before anything is
 * spent. The projection is the only thing standing between a mistyped cohort
 * and a real provider bill, so it refreshes on every change to the selection,
 * the task, or the overrides.
 */

import { Fragment, useMemo, useState } from 'react'
import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { AlertTriangle, ExternalLink, Play, Users } from 'lucide-react'
import {
  Badge,
  Button,
  Callout,
  Card,
  EmptyState,
  Field,
  Skeleton,
  SliderField,
  Toggle,
} from '@/components/ui/primitives'
import { api, ApiError } from '@/lib/api'
import { number, percent, titleCase, tokens, usd } from '@/lib/format'
import { queryKeys, useCapabilities, useDebounced, useParticipants, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { EstimateResponse, Participant } from '@/lib/types'
import { describeRequirements, missingRequirements } from './util'

export function BatchComposer({ onLaunched }: { onLaunched: (batchId: string) => void }) {
  const client = useQueryClient()
  const participantsQuery = useParticipants()
  const settings = useSettings()
  const capabilities = useCapabilities()

  const selected = useApp((s) => s.selected)
  const setSelected = useApp((s) => s.setSelected)
  const toggleSelected = useApp((s) => s.toggleSelected)
  const task = useApp((s) => s.task)
  const overrides = useApp((s) => s.overrides)
  const generateDeepReport = useApp((s) => s.generateDeepReport)
  const setGenerateDeepReport = useApp((s) => s.setGenerateDeepReport)
  const notify = useApp((s) => s.notify)
  const openSettings = useApp((s) => s.openSettings)

  const config = settings.data?.config
  const [concurrency, setConcurrency] = useState<number | null>(null)
  const [continueOnError, setContinueOnError] = useState<boolean | null>(null)
  const [launching, setLaunching] = useState(false)

  const effectiveConcurrency = concurrency ?? config?.batch.concurrency ?? 2
  const effectiveContinue = continueOnError ?? config?.batch.continue_on_error ?? true
  const workers = config?.engine.executor_max_workers ?? 1

  const participants = useMemo(() => participantsQuery.data?.participants ?? [], [participantsQuery.data])
  const selectedDirs = useMemo(() => new Set(selected.map((p) => p.directory)), [selected])

  // Launch only what the current scan still knows about, so a selection left
  // over from a removed data root cannot reach the backend.
  const chosen = useMemo(
    () => participants.filter((p) => p.valid && selectedDirs.has(p.directory)),
    [participants, selectedDirs],
  )
  const valid = useMemo(() => participants.filter((p) => p.valid), [participants])

  const estimateInput = useMemo(
    () => ({
      participant_dirs: chosen.map((p) => p.directory),
      task,
      overrides,
      generate_deep_phenotype: generateDeepReport,
    }),
    [chosen, task, overrides, generateDeepReport],
  )
  const debounced = useDebounced(estimateInput, 400)
  const estimate = useQuery({
    queryKey: ['batch.estimate', JSON.stringify(debounced)],
    queryFn: () => api.runs.estimate(debounced),
    enabled: debounced.participant_dirs.length > 0,
    retry: false,
    staleTime: 60_000,
  })

  const guards = estimate.data?.guards
  const unmet = missingRequirements(task, capabilities.data?.prediction_types)
  const blocked = Boolean(guards?.blocks)
  const canLaunch = chosen.length > 0 && unmet.length === 0 && !blocked && !launching

  const launch = async () => {
    setLaunching(true)
    try {
      const batch = await api.batches.create({
        participant_dirs: chosen.map((p) => p.directory),
        task,
        overrides,
        generate_deep_phenotype: generateDeepReport,
        concurrency: effectiveConcurrency,
        continue_on_error: effectiveContinue,
      })
      client.invalidateQueries({ queryKey: queryKeys.batches })
      client.invalidateQueries({ queryKey: queryKeys.runs })
      notify({
        tone: 'positive',
        title: 'Batch started',
        body: `${batch.total} participant${batch.total === 1 ? '' : 's'} at concurrency ${batch.concurrency}.`,
      })
      onLaunched(batch.id)
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'The batch did not start',
        body: error instanceof ApiError ? error.message : String(error),
      })
    } finally {
      setLaunching(false)
    }
  }

  return (
    <div className="bt__layout">
      <div className="stack gap-5">
        <Card
          title="Cohort"
          subtitle={`${chosen.length} of ${valid.length} valid participant${valid.length === 1 ? '' : 's'} selected`}
          actions={
            valid.length > 0 ? (
              <>
                <Button size="sm" variant="ghost" onClick={() => setSelected(valid)}>
                  All
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setSelected([])}>
                  None
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelected(valid.filter((p) => !selectedDirs.has(p.directory)))}
                >
                  Invert
                </Button>
              </>
            ) : undefined
          }
          flush
        >
          {participantsQuery.isLoading && (
            <div className="stack gap-2" style={{ padding: 'var(--s-5)' }}>
              <Skeleton height={18} />
              <Skeleton height={18} />
              <Skeleton height={18} />
            </div>
          )}

          {!participantsQuery.isLoading && participants.length === 0 && (
            <EmptyState
              icon={<Users size={20} />}
              title="No participants found"
              body="A batch needs a cohort. Add a data root and the scan will pick up every participant directory under it."
              action={
                <Button size="sm" variant="primary" onClick={() => openSettings('workspace')}>
                  Add a data root
                </Button>
              }
            />
          )}

          {participants.length > 0 && (
            <div className="bt__picker">
              <table className="table">
                <thead>
                  <tr>
                    <th style={{ width: 34 }}>
                      <span className="sr-only">Selected</span>
                    </th>
                    <th>Participant</th>
                    <th>Domains</th>
                    <th className="num">Input tokens</th>
                    <th>Validity</th>
                  </tr>
                </thead>
                <tbody>
                  {participants.map((participant) => (
                    <ParticipantRow
                      key={participant.directory}
                      participant={participant}
                      checked={selectedDirs.has(participant.directory)}
                      onToggle={() => toggleSelected(participant)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card
          title="Task"
          subtitle="Designed in Configure and applied to every participant in the batch"
          actions={
            <Link className="btn btn--secondary btn--sm" to="/studio">
              Edit in Configure
            </Link>
          }
        >
          <TaskSummary />
          {unmet.length > 0 && (
            <div style={{ marginTop: 'var(--s-4)' }}>
              <Callout tone="caution" icon={<AlertTriangle size={15} />} title="The task is not complete">
                This task shape still needs {describeRequirements(unmet)}. Set it in Configure and the launch unlocks.
              </Callout>
            </div>
          )}
        </Card>
      </div>

      <div className="stack gap-5">
        <Card title="Execution">
          <div className="stack gap-5">
            <Field
              label="Participants in parallel"
              hint={`Up to about ${effectiveConcurrency * workers} model calls in flight across the batch.`}
              info={
                <>
                  <p>
                    This slider sets how many participants run at the same time. It is not the only source of
                    parallelism: inside each participant the executor fans its plan steps out across up to{' '}
                    {number(workers)} workers.
                  </p>
                  <p>
                    The concurrency the provider actually sees is roughly the product of the two, so {effectiveConcurrency}{' '}
                    participants times {number(workers)} workers is about {effectiveConcurrency * workers} requests in
                    flight. Raise this only if the provider rate limit has room for it.
                  </p>
                </>
              }
            >
              <SliderField
                value={effectiveConcurrency}
                min={1}
                max={16}
                onChange={setConcurrency}
                presets={[1, 2, 4, 8, 16]}
              />
            </Field>

            <div className="row gap-3 between">
              <div className="stack" style={{ gap: 1 }}>
                <span className="t-small semibold">Continue on error</span>
                <span className="t-tiny muted">
                  {effectiveContinue
                    ? 'A failed participant does not stop the rest.'
                    : 'The first failure cancels everything still queued.'}
                </span>
              </div>
              <Toggle checked={effectiveContinue} onChange={setContinueOnError} label="Continue on error" />
            </div>

            <div className="row gap-3 between">
              <div className="stack" style={{ gap: 1 }}>
                <span className="t-small semibold">Deep phenotype report</span>
                <span className="t-tiny muted">
                  {generateDeepReport
                    ? 'The communicator writes a full report for each participant.'
                    : 'Skipped, which removes the largest single cost per participant.'}
                </span>
              </div>
              <Toggle checked={generateDeepReport} onChange={setGenerateDeepReport} label="Deep phenotype report" />
            </div>
          </div>
        </Card>

        <Projection
          query={estimate}
          count={chosen.length}
          onOpenCostSettings={() => openSettings('cost')}
        />

        <Button
          variant="primary"
          size="lg"
          block
          icon={<Play size={15} />}
          disabled={!canLaunch}
          loading={launching}
          onClick={launch}
        >
          {chosen.length === 0
            ? 'Select participants to launch'
            : `Launch ${chosen.length} run${chosen.length === 1 ? '' : 's'}`}
        </Button>
      </div>
    </div>
  )
}

function ParticipantRow({
  participant,
  checked,
  onToggle,
}: {
  participant: Participant
  checked: boolean
  onToggle: () => void
}) {
  const disabled = !participant.valid
  return (
    <tr
      className="bt__row"
      data-selected={checked}
      data-disabled={disabled}
      onClick={disabled ? undefined : onToggle}
    >
      <td>
        <input
          type="checkbox"
          className="bt__check"
          checked={checked}
          disabled={disabled}
          onChange={onToggle}
          onClick={(event) => event.stopPropagation()}
          aria-label={`Select ${participant.id}`}
        />
      </td>
      <td>
        <span className="stack" style={{ gap: 0, minWidth: 0 }}>
          <span className="semibold truncate">{participant.id}</span>
          <span className="t-micro faint truncate">{participant.directory}</span>
        </span>
      </td>
      <td>
        <span className="bt__domains">
          {participant.domains.map((domain) => (
            <Badge key={domain} outline>
              {titleCase(domain)}
            </Badge>
          ))}
          {participant.domains.length === 0 && <span className="faint">none</span>}
        </span>
      </td>
      <td className="num">{tokens(participant.input_tokens)}</td>
      <td>
        {participant.valid ? (
          <Badge tone="positive">Valid</Badge>
        ) : (
          <span className="stack" style={{ gap: 1 }}>
            <Badge tone="critical">Incomplete</Badge>
            <span className="t-micro faint">{participant.missing.join(', ')}</span>
          </span>
        )}
      </td>
    </tr>
  )
}

function TaskSummary() {
  const task = useApp((s) => s.task)
  const capabilities = useCapabilities()
  const spec = capabilities.data?.prediction_types.find((t) => t.value === task.prediction_type)

  const rows: [string, string][] = [
    ['Shape', spec?.label ?? titleCase(task.prediction_type)],
    ['Target', task.target_label || 'not set'],
    ['Comparator', task.control_label || 'not set'],
    ['Classes', task.class_labels.length ? task.class_labels.join(', ') : 'none'],
    ['Outputs', task.regression_outputs.length ? task.regression_outputs.join(', ') : 'none'],
  ]
  if (task.root) rows.push(['Tree root', task.root.display_name || task.root.node_id])

  return (
    <dl className="kv">
      {rows.map(([label, value]) => (
        <Fragment key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

function Projection({
  query,
  count,
  onOpenCostSettings,
}: {
  query: UseQueryResult<EstimateResponse>
  count: number
  onOpenCostSettings: () => void
}) {
  const data = query.data
  const guards = data?.guards

  return (
    <Card
      title="Cost projection"
      subtitle={count > 0 ? `${count} participant${count === 1 ? '' : 's'}, priced before anything runs` : undefined}
      actions={query.isFetching ? <Badge>Updating</Badge> : undefined}
    >
      {count === 0 && <span className="t-small muted">Select at least one participant to see a projection.</span>}

      {count > 0 && query.isLoading && <Skeleton height={90} />}

      {count > 0 && query.error && (
        <Callout tone="critical" icon={<AlertTriangle size={15} />} title="The projection failed">
          {query.error instanceof ApiError ? query.error.message : String(query.error)}
        </Callout>
      )}

      {data && (
        <div className="stack gap-4">
          <div className="bt__totals">
            <div className="stat">
              <span className="stat__label">Total tokens</span>
              <span className="stat__value">{tokens(data.totals.total_tokens)}</span>
            </div>
            <div className="stat">
              <span className="stat__label">Projected cost</span>
              <span className="stat__value">{usd(data.totals.usd)}</span>
              <span className="stat__meta">
                {data.totals.usd_low != null && data.totals.usd_high != null
                  ? `${usd(data.totals.usd_low)} to ${usd(data.totals.usd_high)}`
                  : 'range unavailable'}
              </span>
            </div>
          </div>

          <RangeBar low={data.totals.usd_low} mid={data.totals.usd} high={data.totals.usd_high} />

          {!data.totals.fully_priced && (
            <span className="t-tiny muted">
              At least one model has no published price, so the total is a floor rather than a projection.
            </span>
          )}

          {guards?.blocks && (
            <Callout
              tone="critical"
              icon={<AlertTriangle size={15} />}
              title="Above the hard stop"
              action={
                <Button size="sm" onClick={onOpenCostSettings} icon={<ExternalLink size={13} />}>
                  Cost limits
                </Button>
              }
            >
              The projection of {usd(data.totals.usd)} is above the configured block at {usd(guards.block_above_usd)}, so
              the launch is disabled. Reduce the cohort, turn off the deep report, or raise the limit in Settings under
              Cost.
            </Callout>
          )}

          {guards?.warns && !guards.blocks && (
            <Callout
              tone="caution"
              icon={<AlertTriangle size={15} />}
              title="Above the warning threshold"
              action={
                <Button size="sm" onClick={onOpenCostSettings} icon={<ExternalLink size={13} />}>
                  Cost limits
                </Button>
              }
            >
              This batch is projected at {usd(data.totals.usd)}, above the warning set at {usd(guards.warn_above_usd)}.
              It will still run.
            </Callout>
          )}

          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Participant</th>
                  <th className="num">Tokens</th>
                  <th className="num">Cost</th>
                  <th className="num">Budget</th>
                </tr>
              </thead>
              <tbody>
                {data.participants.map((row) => (
                  <tr key={row.participant_dir}>
                    <td className="truncate">{row.id}</td>
                    <td className="num">{tokens(row.total_tokens)}</td>
                    <td className="num">{usd(row.usd)}</td>
                    <td className="num">
                      {row.budget_utilisation == null ? '-' : percent(row.budget_utilisation, 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  )
}

/** Low to high band with the point estimate marked, drawn on a 0 to high scale. */
function RangeBar({ low, mid, high }: { low: number | null; mid: number | null; high: number | null }) {
  if (low == null || mid == null || high == null || high <= 0) return null
  const start = Math.max(0, Math.min(100, (low / high) * 100))
  const point = Math.max(0, Math.min(100, (mid / high) * 100))
  return (
    <div className="bt__range">
      <span className="bt__range-band" style={{ left: `${start}%`, right: '0%' }} />
      <span className="bt__range-mark" style={{ left: `calc(${point}% - 1px)` }} />
    </div>
  )
}
