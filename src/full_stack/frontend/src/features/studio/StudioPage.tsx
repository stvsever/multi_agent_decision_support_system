/**
 * The configure screen.
 *
 * Three decisions get a run started: whose data, what question, and which
 * model. Everything else has a working default and lives behind a disclosure,
 * so the page reads as three steps rather than a control panel. Settings that
 * matter at launch time are shown here next to the run, with the saved value
 * named, rather than hidden behind a second trip into the settings sheet.
 */

import clsx from 'clsx'
import {
  AlertTriangle,
  Check,
  Cpu,
  Database,
  FlaskConical,
  Layers,
  Play,
  RotateCcw,
  Settings2,
  Target,
  WifiOff,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '@/lib/api'
import { compactNumber, tokens, usd } from '@/lib/format'
import { useCapabilities, useConnectivity, useDebounced, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { EstimateResponse, ReasoningEffort } from '@/lib/types'
import {
  Badge,
  Button,
  Callout,
  Card,
  Disclosure,
  Field,
  InfoDot,
  Progress,
  Segmented,
  SliderField,
  Textarea,
  Toggle,
} from '@/components/ui/primitives'
import { ModelPicker } from '@/components/ui/ModelPicker'
import { BatchOptions, type BatchSettings } from '@/features/batch/BatchOptions'
import { ParticipantPicker } from './ParticipantPicker'
import { TaskDesigner } from './TaskDesigner'
import { taskReport } from './taskValidation'
import './studio.css'

const REASONING_CHOICES: { value: ReasoningEffort | 'saved'; label: string }[] = [
  { value: 'saved', label: 'Saved' },
  { value: 'off', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

const REASONING_LABELS: Record<ReasoningEffort, string> = {
  provider_default: 'provider default',
  off: 'off',
  low: 'low',
  medium: 'medium',
  high: 'high',
}

export function StudioPage() {
  const navigate = useNavigate()
  const { data: capabilities } = useCapabilities()
  const { data: settings } = useSettings()
  const connectivity = useConnectivity()
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
  const [auditing, setAuditing] = useState(false)
  const [advanced, setAdvanced] = useState(settings?.config.appearance.show_advanced_by_default ?? false)
  const [batchOptions, setBatchOptions] = useState<BatchSettings | null>(null)

  const config = settings?.config
  const report = taskReport(task)
  const ready = selected.length > 0 && report.ready
  const isBatch = selected.length > 1

  const batchSettings: BatchSettings = batchOptions ?? {
    concurrency: config?.batch.concurrency ?? 2,
    continueOnError: config?.batch.continue_on_error ?? true,
  }

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

  const models = (overrides.models ?? {}) as Record<string, unknown>
  const engine = (overrides.engine ?? {}) as Record<string, unknown>
  const instructions = (overrides.instructions ?? {}) as Record<string, string>

  const savedModel = config?.models.default_model || capabilities?.default_model || ''
  const modelOverride = (models.default_model as string) || ''
  const effectiveModel = modelOverride || savedModel

  const savedReasoning = config?.models.reasoning_effort ?? 'off'
  const reasoningOverride = models.reasoning_effort as ReasoningEffort | undefined
  const reasoning = reasoningOverride ?? savedReasoning

  const savedIterations = config?.engine.max_iterations ?? 3
  const iterations = (engine.max_iterations as number | undefined) ?? savedIterations
  const savedWorkers = config?.engine.executor_max_workers ?? 12
  const workers = (engine.executor_max_workers as number | undefined) ?? savedWorkers

  const activeOverrides = useMemo(() => {
    const names: string[] = []
    if (modelOverride) names.push('model')
    if (reasoningOverride) names.push('reasoning')
    if (engine.max_iterations !== undefined) names.push('iterations')
    if (engine.executor_max_workers !== undefined) names.push('workers')
    if ((instructions.global ?? '').trim()) names.push('instruction')
    return names
  }, [modelOverride, reasoningOverride, engine.max_iterations, engine.executor_max_workers, instructions.global])

  const launch = async () => {
    if (!ready) return
    setLaunching(true)
    try {
      if (selected.length === 1) {
        const run = await api.runs.create({
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
          concurrency: batchSettings.concurrency,
          continue_on_error: batchSettings.continueOnError,
        })
        notify({
          tone: 'positive',
          title: 'Batch started',
          body: `${batch.total} participants queued, ${batch.concurrency} at a time.`,
        })
        navigate('/batch', { state: { batchId: batch.id } })
      }
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Run could not start',
        body: error instanceof ApiError ? error.message : String(error),
      })
    } finally {
      setLaunching(false)
    }
  }

  const audit = async () => {
    if (selected.length !== 1 || !report.ready) return
    setAuditing(true)
    try {
      const run = await api.runs.audit({ participant_dir: selected[0].directory, task })
      navigate(`/runs/${run.id}`)
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Audit could not start',
        body: error instanceof ApiError ? error.message : String(error),
      })
    } finally {
      setAuditing(false)
    }
  }

  const totals = estimate?.totals
  const guards = estimate?.guards
  const offline = connectivity.data && (!connectivity.data.online || !connectivity.data.provider_reachable)

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
                ? 'none selected'
                : selected.length === 1
                  ? selected[0].id
                  : `${selected.length} selected, this will run as a batch`
            }
            tour="studio-participants"
          >
            <ParticipantPicker multiple />

            {isBatch && (
              <section className="studio__batch">
                <header className="row gap-2">
                  <Layers size={15} />
                  <span className="t-small semibold grow">Batch controls</span>
                  <Badge tone="accent">{selected.length} participants</Badge>
                </header>
                <BatchOptions
                  value={batchSettings}
                  onChange={setBatchOptions}
                  participantCount={selected.length}
                />
              </section>
            )}
          </Step>

          <Step
            index={2}
            icon={<Target size={15} />}
            title="Prediction task"
            done={report.ready}
            summary={report.summary}
            tour="studio-task"
          >
            <TaskDesigner
              task={task}
              report={report}
              onChange={setTask}
              types={capabilities?.prediction_types ?? []}
              onNotify={(title, body) => notify({ tone: 'positive', title, body })}
            />
          </Step>

          <Step
            index={3}
            icon={<Cpu size={15} />}
            title="Model and engine"
            done
            summary={`${effectiveModel || 'no model'}, reasoning ${REASONING_LABELS[reasoning]}, ${iterations} iteration${iterations === 1 ? '' : 's'}`}
            tour="studio-model"
          >
            <div className="stack gap-5">
              <div className="row between gap-3 wrap">
                <div className="stack gap-1" style={{ minWidth: 0 }}>
                  <span className="eyebrow">Model in use</span>
                  <span className="row gap-2 wrap">
                    <span className="mono semibold">{effectiveModel || 'not set'}</span>
                    {modelOverride && <Badge tone="accent">this run only</Badge>}
                  </span>
                  <span className="t-tiny muted">
                    {modelOverride
                      ? `The saved default is ${savedModel || 'not set'}. It is untouched.`
                      : 'Applies to every agent role unless a role override is set in settings.'}
                  </span>
                </div>
                <Button size="sm" icon={<Settings2 size={13} />} onClick={() => openSettings('compute')}>
                  Change models
                </Button>
              </div>

              <div className="grid grid--2">
                <Field
                  label="Model for this run"
                  hint="Searches the live provider catalog. Leaving it on the saved default is the usual choice."
                >
                  <ModelPicker
                    value={modelOverride}
                    onChange={(default_model) => setOverride('models', { default_model: default_model || undefined })}
                    allowInherit
                    inheritLabel={savedModel ? `Saved default: ${savedModel}` : 'Saved default'}
                  />
                </Field>

                <Field
                  label="Reasoning effort"
                  info={
                    <>
                      <p>
                        Reasoning tokens are billed as output and count against the output ceiling. Turning
                        reasoning off is markedly faster and cheaper; raise it when the task needs deeper
                        deliberation.
                      </p>
                      <p>
                        The saved value applies to every run and lives in settings. Choosing anything else here
                        changes this run only.
                      </p>
                    </>
                  }
                  hint={
                    reasoningOverride
                      ? `Overridden for this run. The saved value is ${REASONING_LABELS[savedReasoning]}.`
                      : `Using the saved value, ${REASONING_LABELS[savedReasoning]}.`
                  }
                >
                  <Segmented
                    block
                    value={(reasoningOverride ?? 'saved') as ReasoningEffort | 'saved'}
                    options={REASONING_CHOICES.map((choice) =>
                      choice.value === 'saved' ? { ...choice, label: `Saved: ${REASONING_LABELS[savedReasoning]}` } : choice,
                    )}
                    onChange={(next) =>
                      setOverride('models', {
                        reasoning_effort: next === 'saved' ? undefined : (next as ReasoningEffort),
                      })
                    }
                  />
                </Field>
              </div>

              <div className="grid grid--2">
                <Field
                  label={
                    <OverrideLabel
                      label="Critic iterations"
                      overridden={engine.max_iterations !== undefined}
                      saved={String(savedIterations)}
                      onReset={() => setOverride('engine', { max_iterations: undefined })}
                    />
                  }
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
                    suffix={iterations === 1 ? 'pass' : 'passes'}
                    onChange={(max_iterations) => setOverride('engine', { max_iterations })}
                  />
                </Field>

                <Field
                  label={
                    <OverrideLabel
                      label="Parallel tool workers"
                      overridden={engine.executor_max_workers !== undefined}
                      saved={String(savedWorkers)}
                      onReset={() => setOverride('engine', { executor_max_workers: undefined })}
                    />
                  }
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
                    suffix="at once"
                    onChange={(executor_max_workers) => setOverride('engine', { executor_max_workers })}
                  />
                </Field>
              </div>

              <div className="row between gap-3">
                <div className="stack" style={{ gap: 1 }}>
                  <span className="t-small semibold">Generate the deep phenotype report</span>
                  <span className="t-tiny muted">
                    The Communicator writes an evidence-grounded report and marks missing information explicitly.
                    It is the largest single cost per participant.
                  </span>
                </div>
                <Toggle checked={generateDeepReport} onChange={setGenerateDeepReport} label="Deep phenotype report" />
              </div>

              <Disclosure
                title="Extra instruction for this run"
                subtitle="Appended to every agent prompt, without changing the saved instructions"
                open={advanced}
                onOpenChange={setAdvanced}
                right={(instructions.global ?? '').trim() ? <Badge tone="accent">set</Badge> : undefined}
              >
                <div className="stack gap-4">
                  <Field
                    label="Applies to every agent"
                    hint="Use it for study-specific framing. Per-agent instructions and token budgets are saved settings."
                  >
                    <Textarea
                      rows={3}
                      placeholder="for example: this cohort is medication naive; do not infer treatment effects."
                      value={instructions.global ?? ''}
                      onChange={(event) => setOverride('instructions', { ...instructions, global: event.target.value })}
                    />
                  </Field>
                  <div className="row gap-2">
                    <Button size="sm" variant="ghost" onClick={() => openSettings('instructions')}>
                      Saved instructions
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => openSettings('engine')}>
                      Token budgets
                    </Button>
                  </div>
                </div>
              </Disclosure>

              {activeOverrides.length > 0 && (
                <div className="row between gap-3 wrap">
                  <span className="t-tiny muted">
                    This run overrides {activeOverrides.join(', ')}. The saved configuration is untouched.
                  </span>
                  <Button size="sm" variant="ghost" icon={<RotateCcw size={13} />} onClick={clearOverrides}>
                    Reset to saved
                  </Button>
                </div>
              )}
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

          <div data-tour="studio-launch">
            <Card title="Ready to run">
              <div className="stack gap-3">
                <ChecklistRow
                  ok={selected.length > 0}
                  label={
                    selected.length > 1
                      ? `${selected.length} participants selected`
                      : selected.length === 1
                        ? '1 participant selected'
                        : 'Participants selected'
                  }
                />
                <ChecklistRow ok={report.ready} label={report.ready ? 'Task is valid' : `Task ${report.summary}`} />
                <ChecklistRow ok={Boolean(effectiveModel)} label="Model chosen" />

                {offline && (
                  <Callout tone="caution" icon={<WifiOff size={15} />} title="The provider is not reachable">
                    {connectivity.data?.reason || 'A run started now would fail on its first call.'}
                  </Callout>
                )}

                <Button
                  variant="primary"
                  size="lg"
                  block
                  icon={<Play size={15} />}
                  disabled={!ready || Boolean(guards?.blocks)}
                  loading={launching}
                  onClick={launch}
                >
                  {isBatch ? `Run ${selected.length} participants` : 'Run the pipeline'}
                </Button>

                <div className="stack gap-2">
                  <Button
                    block
                    icon={<FlaskConical size={14} />}
                    disabled={selected.length !== 1 || !report.ready}
                    loading={auditing}
                    onClick={audit}
                  >
                    Dry run: structural audit
                  </Button>
                  <span className="t-tiny muted">
                    {selected.length > 1
                      ? 'An audit reads one participant at a time. Select a single participant to audit.'
                      : 'Loads the inputs and builds the predictor payload, then reports token counts, chunking, and the coverage assertions. No provider is called, so it is free and returns no prediction.'}
                  </span>
                </div>

                {estimating && <Progress indeterminate />}
              </div>
            </Card>
          </div>

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
          <span className={clsx('t-tiny truncate', done ? 'muted' : 'studio__step-todo')}>{summary}</span>
        </span>
      </header>
      <div className="studio__step-body">{children}</div>
    </section>
  )
}

/** A label that says when the control has been moved off its saved value. */
function OverrideLabel({
  label,
  overridden,
  saved,
  onReset,
}: {
  label: string
  overridden: boolean
  saved: string
  onReset: () => void
}) {
  return (
    <span className="row gap-2">
      <span>{label}</span>
      {overridden && (
        <>
          <Badge tone="accent">this run</Badge>
          <button type="button" className="studio__reset" onClick={onReset}>
            back to {saved}
          </button>
        </>
      )}
    </span>
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
