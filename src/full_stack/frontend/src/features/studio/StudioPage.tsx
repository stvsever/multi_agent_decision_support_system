/**
 * The configure screen.
 *
 * Three decisions get a run started: whose data, what question, and which
 * model. Everything else has a working default and lives behind a disclosure,
 * so the page reads as three steps rather than a control panel.
 */

import clsx from 'clsx'
import {
  AlertTriangle,
  Check,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  Play,
  Settings2,
  Target,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '@/lib/api'
import { compactNumber, tokens, usd } from '@/lib/format'
import { useCapabilities, useDebounced, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { EstimateResponse, RunOverrides } from '@/lib/types'
import {
  Badge,
  Button,
  Callout,
  Card,
  Disclosure,
  Field,
  InfoDot,
  Input,
  Progress,
  Segmented,
  SliderField,
  Toggle,
  Tooltip,
} from '@/components/ui/primitives'
import { ParticipantPicker } from './ParticipantPicker'
import { TaskDesigner, validateTask } from './TaskDesigner'
import './studio.css'

export function StudioPage() {
  const navigate = useNavigate()
  const { data: capabilities } = useCapabilities()
  const { data: settings } = useSettings()
  const {
    selected,
    task,
    setTask,
    overrides,
    setOverride,
    clearOverrides,
    generateDeepReport,
    setGenerateDeepReport,
    notify,
    openSettings,
  } = useApp()

  const [estimate, setEstimate] = useState<EstimateResponse | null>(null)
  const [estimating, setEstimating] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [advanced, setAdvanced] = useState(settings?.config.appearance.show_advanced_by_default ?? false)

  const config = settings?.config
  const problems = validateTask(task)
  const ready = selected.length > 0 && problems.length === 0

  // The projection depends on the selection, the task, and any per-run deltas,
  // so it is recomputed as those settle rather than on every keystroke.
  const estimateKey = useDebounced(
    JSON.stringify({ dirs: selected.map((s) => s.directory), task, overrides, generateDeepReport }),
    400,
  )

  useEffect(() => {
    if (selected.length === 0) {
      setEstimate(null)
      return
    }
    let cancelled = false
    setEstimating(true)
    api.runs
      .estimate({
        participant_dirs: selected.map((s) => s.directory),
        task,
        overrides,
        generate_deep_phenotype: generateDeepReport,
      })
      .then((result) => {
        if (!cancelled) setEstimate(result)
      })
      .catch(() => {
        if (!cancelled) setEstimate(null)
      })
      .finally(() => {
        if (!cancelled) setEstimating(false)
      })
    return () => {
      cancelled = true
    }
    // estimateKey encodes every input the projection depends on.
  }, [estimateKey])

  const effectiveModel = useMemo(() => {
    const override = (overrides.models as { default_model?: string } | undefined)?.default_model
    return override || config?.models.default_model || capabilities?.default_model || ''
  }, [overrides.models, config?.models.default_model, capabilities?.default_model])

  const iterations =
    (overrides.engine as { max_iterations?: number } | undefined)?.max_iterations ??
    config?.engine.max_iterations ??
    3
  const workers =
    (overrides.engine as { executor_max_workers?: number } | undefined)?.executor_max_workers ??
    config?.engine.executor_max_workers ??
    12
  const reasoning =
    (overrides.models as { reasoning_effort?: string } | undefined)?.reasoning_effort ??
    config?.models.reasoning_effort ??
    'off'

  const launch = async (audit = false) => {
    if (!ready) return
    setLaunching(true)
    try {
      if (selected.length === 1) {
        const run = audit
          ? await api.runs.audit({ participant_dir: selected[0].directory, task })
          : await api.runs.create({
              participant_dir: selected[0].directory,
              task,
              overrides,
              generate_deep_phenotype: generateDeepReport,
            })
        navigate(`/runs/${run.id}`)
      } else {
        const batch = await api.batches.create({
          participant_dirs: selected.map((s) => s.directory),
          task,
          overrides,
          generate_deep_phenotype: generateDeepReport,
        })
        notify({ tone: 'positive', title: 'Batch started', body: `${batch.total} participants queued.` })
        navigate('/batch')
      }
    } catch (error) {
      notify({
        tone: 'critical',
        title: audit ? 'Audit could not start' : 'Run could not start',
        body: error instanceof ApiError ? error.message : String(error),
      })
    } finally {
      setLaunching(false)
    }
  }

  const totals = estimate?.totals
  const guards = estimate?.guards

  return (
    <div className="page studio">
      <header className="page__head">
        <div className="stack gap-2">
          <span className="page__title">Configure a run</span>
          <p className="page__lede">
            Pick whose data to read, state the question, and confirm the model. Everything else has a working
            default, and the projected cost updates as you go.
          </p>
        </div>
      </header>

      <div className="studio__grid">
        <div className="stack gap-4" style={{ minWidth: 0 }}>
          <Step
            index={1}
            icon={<Database size={15} />}
            title="Participants"
            done={selected.length > 0}
            summary={
              selected.length === 0
                ? 'None selected'
                : selected.length === 1
                  ? selected[0].id
                  : `${selected.length} selected`
            }
            tour="studio-participants"
          >
            <ParticipantPicker multiple />
            {selected.length > 1 && (
              <Callout tone="info" title={`${selected.length} participants selected`}>
                Launching will queue a batch. Concurrency and failure handling are set under Batch in settings.
              </Callout>
            )}
          </Step>

          <Step
            index={2}
            icon={<Target size={15} />}
            title="Prediction task"
            done={problems.length === 0 && Boolean(task.target_label.trim() || task.root)}
            summary={
              problems.length === 0
                ? `${capabilities?.prediction_types.find((t) => t.value === task.prediction_type)?.label ?? task.prediction_type}: ${task.target_label || task.root?.display_name || 'unnamed'}`
                : `${problems.length} thing${problems.length === 1 ? '' : 's'} to fix`
            }
            tour="studio-task"
          >
            <TaskDesigner task={task} onChange={setTask} types={capabilities?.prediction_types ?? []} />
          </Step>

          <Step
            index={3}
            icon={<Cpu size={15} />}
            title="Model and engine"
            done
            summary={`${effectiveModel || 'no model'} · ${iterations} iteration${iterations === 1 ? '' : 's'}`}
            tour="studio-model"
          >
            <div className="stack gap-4">
              <div className="row between gap-3 wrap">
                <div className="stack gap-1" style={{ minWidth: 0 }}>
                  <span className="eyebrow">Model in use</span>
                  <span className="row gap-2 wrap">
                    <span className="mono semibold">{effectiveModel || 'not set'}</span>
                    <Badge tone={reasoning === 'off' ? 'neutral' : 'accent'}>reasoning {reasoning}</Badge>
                  </span>
                  <span className="t-tiny muted">
                    Applies to every agent role unless a role override is set in settings.
                  </span>
                </div>
                <Button size="sm" icon={<Settings2 size={13} />} onClick={() => openSettings('models')}>
                  Change models
                </Button>
              </div>

              <div className="grid grid--2">
                <Field
                  label="Critic iterations"
                  info={
                    <p>
                      Each iteration replans, re-executes, re-predicts, and re-scores. The loop stops early when
                      the Critic returns a satisfactory verdict, so a higher ceiling costs nothing when the first
                      attempt is good.
                    </p>
                  }
                >
                  <SliderField
                    value={iterations}
                    min={1}
                    max={10}
                    presets={[1, 2, 3, 5]}
                    onChange={(max_iterations) => setOverride('engine', { max_iterations })}
                  />
                </Field>

                <Field
                  label="Parallel tool workers"
                  info={
                    <p>
                      The executor runs every unblocked plan step at once, bounded by this number. Raising it
                      shortens the execution stage but increases concurrent provider calls, which can hit a rate
                      limit.
                    </p>
                  }
                >
                  <SliderField
                    value={workers}
                    min={1}
                    max={64}
                    presets={[1, 4, 8, 12, 24]}
                    onChange={(executor_max_workers) => setOverride('engine', { executor_max_workers })}
                  />
                </Field>
              </div>

              <div className="row between gap-3">
                <Field
                  label="Generate the deep phenotype report"
                  hint="The Communicator writes an evidence-grounded report and marks missing information explicitly."
                >
                  <div />
                </Field>
                <Toggle checked={generateDeepReport} onChange={setGenerateDeepReport} label="Deep phenotype report" />
              </div>

              <Disclosure
                title="Advanced overrides for this run"
                subtitle="Applied on top of the saved configuration, without changing it"
                open={advanced}
                onOpenChange={setAdvanced}
                right={
                  Object.keys(overrides).length > 0 ? (
                    <Badge tone="accent">{Object.keys(overrides).length} section</Badge>
                  ) : undefined
                }
              >
                <RunOverridePanel overrides={overrides} setOverride={setOverride} onClear={clearOverrides} />
              </Disclosure>
            </div>
          </Step>
        </div>

        <aside className="studio__rail">
          <Card title="Projected cost" subtitle={estimating ? 'Recalculating' : 'From live provider pricing'}>
            {selected.length === 0 ? (
              <p className="t-small muted">Select at least one participant to see a projection.</p>
            ) : !totals ? (
              <p className="t-small muted">
                The projection needs the model catalog. Check the provider connection in settings.
              </p>
            ) : (
              <div className="stack gap-4">
                <div className="stack gap-1">
                  <span className="stat__value" style={{ fontSize: 'var(--t-display)' }}>
                    {totals.fully_priced ? usd(totals.usd) : 'n/a'}
                  </span>
                  {totals.fully_priced && (
                    <span className="t-tiny muted tabular">
                      likely {usd(totals.usd_low)} to {usd(totals.usd_high)}
                    </span>
                  )}
                  <span className="t-tiny muted tabular">
                    {tokens(totals.total_tokens)} tokens across {totals.count} participant
                    {totals.count === 1 ? '' : 's'}
                  </span>
                </div>

                {estimate && estimate.participants.length > 0 && (
                  <div className="stack gap-2">
                    <span className="eyebrow row gap-2">
                      Per agent
                      <InfoDot>
                        <p>
                          The projection is a linear model fitted against the token ledgers of completed runs. It
                          scales with the evidence volume in the participant's own data overview, so it moves with
                          the data rather than being a flat guess.
                        </p>
                        <p>
                          Plan size is genuinely variable, which is why a range is shown alongside the point
                          estimate.
                        </p>
                      </InfoDot>
                    </span>
                    {estimate.participants[0].lines.map((line) => (
                      <div key={line.role} className="row gap-2 t-tiny">
                        <span className="grow truncate" style={{ textTransform: 'capitalize' }}>
                          {line.role}
                        </span>
                        <span className="muted tabular">{compactNumber(line.total_tokens)}</span>
                        <span className="tabular" style={{ width: 62, textAlign: 'right' }}>
                          {line.usd === null ? 'n/a' : usd(line.usd)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {guards?.blocks && (
                  <Callout tone="critical" icon={<AlertTriangle size={15} />} title="Above the hard stop">
                    This projection exceeds the configured limit of {usd(guards.block_above_usd)}. Raise it under
                    Cost guardrails to proceed.
                  </Callout>
                )}
                {guards?.warns && !guards.blocks && (
                  <Callout tone="caution" icon={<AlertTriangle size={15} />} title="Above your warning threshold">
                    You set a warning at {usd(guards.warn_above_usd)}.
                  </Callout>
                )}
              </div>
            )}
          </Card>

          <Card title="Ready to run" data-tour="studio-launch">
            <div className="stack gap-3">
              <ChecklistRow ok={selected.length > 0} label="Participants selected" />
              <ChecklistRow ok={problems.length === 0} label="Task is valid" />
              <ChecklistRow ok={Boolean(effectiveModel)} label="Model chosen" />

              <Button
                variant="primary"
                size="lg"
                block
                icon={<Play size={15} />}
                disabled={!ready || Boolean(guards?.blocks)}
                loading={launching}
                onClick={() => launch(false)}
              >
                {selected.length > 1 ? `Run ${selected.length} participants` : 'Run the pipeline'}
              </Button>

              <Tooltip content="Loads the data and builds the predictor payload without contacting any provider. Free.">
                <Button
                  block
                  icon={<FlaskConical size={14} />}
                  disabled={selected.length !== 1 || problems.length > 0}
                  onClick={() => launch(true)}
                >
                  Dry run: structural audit
                </Button>
              </Tooltip>

              {estimating && <Progress indeterminate />}
            </div>
          </Card>

          <Card title="What happens next">
            <ol className="studio__steps">
              {(capabilities?.agents ?? []).map((agent) => (
                <li key={agent.role}>
                  <span className="studio__steps-dot" />
                  <span className="stack gap-1">
                    <span className="semibold t-small">{agent.label}</span>
                    <span className="t-tiny muted">{agent.summary}</span>
                  </span>
                </li>
              ))}
            </ol>
          </Card>
        </aside>
      </div>
    </div>
  )
}

/* --- Pieces --------------------------------------------------------------- */

function Step({
  index,
  icon,
  title,
  summary,
  done,
  children,
  tour,
}: {
  index: number
  icon: React.ReactNode
  title: string
  summary: string
  done: boolean
  children: React.ReactNode
  tour?: string
}) {
  return (
    <section className="studio__step" data-tour={tour}>
      <header className="studio__step-head">
        <span className={clsx('studio__step-num', done && 'studio__step-num--done')}>
          {done ? <Check size={14} /> : index}
        </span>
        <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
          <span className="row gap-2">
            {icon}
            <span className="t-h3">{title}</span>
          </span>
          <span className="t-tiny muted truncate">{summary}</span>
        </span>
      </header>
      <div className="studio__step-body">{children}</div>
    </section>
  )
}

function ChecklistRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="row gap-2 t-small">
      <span
        className="row center"
        style={{
          width: 16,
          height: 16,
          borderRadius: '50%',
          flex: 'none',
          background: ok ? 'var(--positive-soft)' : 'var(--bg-inset)',
          color: ok ? 'var(--positive)' : 'var(--text-faint)',
        }}
      >
        {ok ? <Check size={11} /> : <span style={{ fontSize: 9 }}>·</span>}
      </span>
      <span className={ok ? undefined : 'muted'}>{label}</span>
    </div>
  )
}

function RunOverridePanel({
  overrides,
  setOverride,
  onClear,
}: {
  overrides: RunOverrides
  setOverride: (section: keyof RunOverrides, patch: Record<string, unknown>) => void
  onClear: () => void
}) {
  const models = (overrides.models ?? {}) as Record<string, unknown>
  const budget = (overrides.token_budget ?? {}) as Record<string, unknown>
  const instructions = (overrides.instructions ?? {}) as Record<string, string>

  return (
    <div className="stack gap-4">
      <Callout tone="neutral" icon={<Gauge size={15} />}>
        These apply to this run only. The saved configuration is left untouched.
      </Callout>

      <div className="grid grid--2">
        <Field label="Model for this run" hint="Leave blank to use the saved default.">
          <Input
            mono
            placeholder="provider/model-id"
            value={(models.default_model as string) ?? ''}
            onChange={(e) => setOverride('models', { default_model: e.target.value })}
          />
        </Field>
        <Field
          label="Reasoning effort"
          info={
            <p>
              Reasoning tokens are billed as output and count against the output ceiling. Turning reasoning off
              is markedly faster and cheaper; raise it when the task needs deeper deliberation.
            </p>
          }
        >
          <Segmented
            value={(models.reasoning_effort as string) ?? 'off'}
            options={[
              { value: 'off', label: 'Off' },
              { value: 'low', label: 'Low' },
              { value: 'medium', label: 'Medium' },
              { value: 'high', label: 'High' },
            ]}
            onChange={(reasoning_effort) => setOverride('models', { reasoning_effort })}
          />
        </Field>
        <Field label="Total token budget" hint="0 keeps the saved value.">
          <Input
            type="number"
            min={0}
            value={(budget.total_budget as number) ?? ''}
            onChange={(e) => setOverride('token_budget', { total_budget: Number(e.target.value) || 0 })}
          />
        </Field>
        <Field label="Max agent output tokens" hint="0 derives from the context window.">
          <Input
            type="number"
            min={0}
            value={(budget.max_agent_output_tokens as number) ?? ''}
            onChange={(e) =>
              setOverride('token_budget', { max_agent_output_tokens: Number(e.target.value) || 0 })
            }
          />
        </Field>
      </div>

      <Field
        label="Extra instruction for every agent"
        info={
          <p>
            Appended to each agent's system prompt for this run. Use it for study-specific framing, for example a
            cohort description or a reporting convention. Per-agent instructions live in settings.
          </p>
        }
      >
        <textarea
          className="textarea"
          rows={3}
          placeholder="for example: this cohort is medication naive; do not infer treatment effects."
          value={instructions.global ?? ''}
          onChange={(e) => setOverride('instructions', { ...instructions, global: e.target.value })}
        />
      </Field>

      {Object.keys(overrides).length > 0 && (
        <div className="row">
          <Button size="sm" variant="ghost" onClick={onClear}>
            Clear all overrides
          </Button>
        </div>
      )}
    </div>
  )
}
