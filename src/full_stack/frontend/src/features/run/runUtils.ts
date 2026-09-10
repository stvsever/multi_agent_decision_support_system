/**
 * Derivations shared by the run list and the live console.
 *
 * The service sends loosely typed payloads (`state.prediction`, `event.data`)
 * because their shape depends on the task. Everything here reads them
 * defensively so a run that has emitted a single event still renders.
 */

import { titleCase } from '@/lib/format'
import type {
  RunDetail,
  RunEvent,
  RunGraph,
  RunStatus,
  RunStep,
  TaskNodeInput,
  TaskSpecInput,
  TaskSummary,
} from '@/lib/types'

/** Fallback for a record that has not carried its stage list yet. */
export const STAGE_NAMES = [
  'Initialization',
  'Orchestration',
  'Execution',
  'Integration',
  'Prediction',
  'Evaluation',
  'Communication',
]

export type Tone = 'neutral' | 'accent' | 'positive' | 'caution' | 'critical' | 'info'

const ACTIVE: RunStatus[] = ['queued', 'running', 'cancelling']

export const isActive = (status: RunStatus): boolean => ACTIVE.includes(status)
export const isTerminal = (status: RunStatus): boolean => !isActive(status)

export function statusTone(status: RunStatus): Tone {
  switch (status) {
    case 'succeeded':
      return 'positive'
    case 'failed':
      return 'critical'
    case 'cancelling':
      return 'caution'
    case 'running':
      return 'accent'
    case 'cancelled':
    case 'queued':
    default:
      return 'neutral'
  }
}

export function verdictTone(verdict: string | undefined | null): Tone {
  const text = String(verdict ?? '').toUpperCase()
  if (!text) return 'neutral'
  if (text.includes('SATISFACTORY') && !text.includes('UNSATISFACTORY')) return 'positive'
  if (text.includes('UNSATISFACTORY') || text.includes('REJECT')) return 'critical'
  if (text.includes('PARTIAL') || text.includes('REVISE')) return 'caution'
  return 'neutral'
}

/* --- Safe readers for loosely typed payloads ------------------------------ */

export function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function asText(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return ''
}

export function asNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function asStringList(value: unknown): string[] {
  return asArray(value)
    .map((item) => asText(item))
    .filter((item) => item.length > 0)
}

/* --- Stage timing --------------------------------------------------------- */

export interface StageTimings {
  /** Seconds spent in each stage, summed across iterations. */
  totals: number[]
  visited: boolean[]
  /** Stage the last timed event put the run into, or -1 before anything ran. */
  current: number
}

/**
 * Event timestamps are naive local ISO strings from the engine process, while
 * `started_at` and `finished_at` are UTC. The two are never mixed: stage
 * accounting diffs event timestamps against each other, and against `nowMs`
 * only while the run is live (the engine runs on this machine).
 */
function eventMs(event: RunEvent): number | null {
  if (!event.ts) return null
  const parsed = Date.parse(event.ts)
  return Number.isNaN(parsed) ? null : parsed
}

/** The stage an event moves the run into, mirroring the service reducer. */
export function eventStage(event: RunEvent, stageCount: number): number | null {
  const data = asRecord(event.data)
  const explicit = data.stage
  if (typeof explicit === 'number' && Number.isFinite(explicit)) return explicit
  switch (event.type) {
    case 'INIT':
      return 0
    case 'PLAN':
      return 1
    case 'FUSION':
      return 3
    case 'PREDICTION':
      return 4
    case 'CRITIC':
      return 5
    case 'COMPLETE':
      return Math.max(0, stageCount - 1)
    case 'STEP_START': {
      const stepId = asNumber(data.id) ?? 0
      if (stepId >= 930) return 6
      if (stepId >= 920) return 5
      if (stepId >= 910) return 4
      if (stepId >= 900) return 3
      return 2
    }
    default:
      return null
  }
}

/**
 * Attribute wall-clock time to stages from the event stream.
 *
 * Every stage-bearing event closes the interval that began at the previous one,
 * so a stage that is entered twice (an extra critic iteration) accumulates both
 * visits. Pass `nowMs` for a live run to keep the open stage counting; pass null
 * and the final interval closes at the last timed event.
 */
export function stageTimings(
  events: RunEvent[],
  stageCount: number,
  nowMs: number | null,
  startedMs: number | null = null,
): StageTimings {
  const totals = new Array<number>(Math.max(0, stageCount)).fill(0)
  const visited = new Array<boolean>(Math.max(0, stageCount)).fill(false)
  let current = -1
  let since = 0

  /*
   * A run is in Initialization from the moment it starts, not from the moment
   * its first event arrives. The worker has to boot a process and import the
   * engine before it can say anything, and reading the rail only from the
   * stream left the first stage blank for those seconds, as though nothing had
   * happened yet. Seeding from the run's own start time makes the rail honest
   * about where the time is going.
   */
  if (startedMs !== null && stageCount > 0) {
    current = 0
    visited[0] = true
    since = startedMs
  }

  for (const event of events) {
    const stage = eventStage(event, stageCount)
    if (stage === null || stage < 0 || stage >= stageCount) continue
    const at = eventMs(event)
    if (at === null) continue
    if (current >= 0 && at > since) totals[current] += (at - since) / 1000
    current = stage
    visited[stage] = true
    since = at
  }

  const end = nowMs ?? since
  if (current >= 0 && end > since) totals[current] += (end - since) / 1000
  return { totals, visited, current }
}

/* --- Steps ---------------------------------------------------------------- */

/** The reducer stamps a wall-clock start on each step; the contract omits it. */
export function stepStartMs(step: RunStep): number | null {
  const raw = (step as unknown as { startTime?: unknown }).startTime
  const seconds = asNumber(raw)
  return seconds === null ? null : seconds * 1000
}

export function stepTone(status: RunStep['status']): Tone {
  switch (status) {
    case 'complete':
      return 'positive'
    case 'failed':
      return 'critical'
    case 'repairing':
      return 'caution'
    default:
      return 'accent'
  }
}

/** step_id to dependency rank, which is exactly the concurrency grouping. */
export function ranksFromGraph(graph: RunGraph | null | undefined): Map<number, number> {
  const ranks = new Map<number, number>()
  for (const node of graph?.nodes ?? []) {
    if (node.type === 'tool' && typeof node.step_id === 'number') ranks.set(node.step_id, node.rank)
  }
  return ranks
}

export interface StepGroup {
  key: string
  rank: number | null
  steps: RunStep[]
}

export interface IterationGroup {
  iteration: number
  groups: StepGroup[]
  steps: RunStep[]
}

const CONCURRENT_WINDOW_MS = 900

/** Ids at or above this are the agent scaffold, which always runs on its own. */
const VIRTUAL_STEP_FLOOR = 900

/**
 * Split steps into the batches that ran together. Plan ranks are authoritative;
 * without a graph, steps that started within the same instant are treated as one
 * dispatch, which is how the executor releases unblocked steps.
 */
export function groupConcurrent(steps: RunStep[], ranks: Map<number, number>): StepGroup[] {
  const groups: StepGroup[] = []
  for (const step of steps) {
    const rank = ranks.get(step.id) ?? null
    const current = groups[groups.length - 1]
    const previous = current?.steps[current.steps.length - 1]
    let joins = false
    if (current && previous && step.id < VIRTUAL_STEP_FLOOR && previous.id < VIRTUAL_STEP_FLOOR) {
      if (rank !== null && current.rank !== null) {
        joins = rank === current.rank
      } else if (rank === null && current.rank === null) {
        const a = stepStartMs(step)
        const b = stepStartMs(previous)
        joins = a !== null && b !== null && Math.abs(a - b) <= CONCURRENT_WINDOW_MS
      }
    }
    if (joins && current) current.steps.push(step)
    else groups.push({ key: `g${step.id}`, rank, steps: [step] })
  }
  return groups
}

export function groupByIteration(steps: RunStep[], ranks: Map<number, number>): IterationGroup[] {
  const order: number[] = []
  const buckets = new Map<number, RunStep[]>()
  for (const step of steps) {
    const iteration = step.iteration ?? 1
    if (!buckets.has(iteration)) {
      buckets.set(iteration, [])
      order.push(iteration)
    }
    buckets.get(iteration)!.push(step)
  }
  return order.map((iteration) => {
    const rows = buckets.get(iteration) ?? []
    return { iteration, steps: rows, groups: groupConcurrent(rows, ranks) }
  })
}

/* --- Events --------------------------------------------------------------- */

export function eventTone(type: string): Tone {
  switch (type) {
    case 'STEP_FAIL':
      return 'critical'
    case 'REPAIR':
      return 'caution'
    case 'COMPLETE':
      return 'positive'
    case 'PREDICTION':
    case 'CRITIC':
      return 'accent'
    case 'PLAN':
    case 'INIT':
      return 'info'
    default:
      return 'neutral'
  }
}

function clip(text: string, limit = 220): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > limit ? `${flat.slice(0, limit - 1)}...` : flat
}

/** One line describing an event's payload, whatever its type. */
export function summariseEvent(event: RunEvent): string {
  const data = asRecord(event.data)
  switch (event.type) {
    case 'STATUS':
      return clip(asText(data.message))
    case 'INIT': {
      const target = asText(data.target)
      const control = asText(data.control)
      const parts = [asText(data.participant_id), control ? `${target} vs ${control}` : target]
      return clip(parts.filter(Boolean).join(' - '))
    }
    case 'PLAN': {
      const steps = asNumber(data.steps)
      const domains = asStringList(data.domains)
      return clip([steps !== null ? `${steps} steps` : '', domains.join(', ')].filter(Boolean).join(' - '))
    }
    case 'STEP_START':
      return clip(`#${asText(data.id)} ${asText(data.tool)}: ${asText(data.desc)}`)
    case 'STEP_COMPLETE': {
      const ms = asNumber(data.duration_ms)
      const head = `#${asText(data.id)} done, ${asText(data.tokens) || '0'} tokens${ms !== null ? `, ${Math.round(ms)}ms` : ''}`
      const preview = asText(data.preview)
      return clip(preview ? `${head} - ${preview}` : head)
    }
    case 'STEP_FAIL':
      return clip(`#${asText(data.id)} ${asText(data.error)}`)
    case 'REPAIR':
      return clip(`#${asText(data.id)} ${asText(data.strategy)}`)
    case 'FUSION': {
      const keys = Object.keys(data)
      return clip(keys.length ? keys.join(', ') : 'Fusion complete')
    }
    case 'PREDICTION': {
      const probability = asNumber(data.prob)
      const label = asText(data.result) || asText(data.label)
      const confidence = asText(data.confidence)
      return clip(
        [label, probability !== null ? `${(probability * 100).toFixed(1)}%` : '', confidence]
          .filter(Boolean)
          .join(' - '),
      )
    }
    case 'CRITIC': {
      const passed = asNumber(data.passed)
      const total = asNumber(data.total)
      const score = asNumber(data.composite_score)
      return clip(
        [
          asText(data.verdict),
          passed !== null && total !== null ? `${passed}/${total} checks` : '',
          score !== null ? `score ${score.toFixed(2)}` : '',
        ]
          .filter(Boolean)
          .join(' - '),
      )
    }
    case 'COMPLETE': {
      const seconds = asNumber(data.duration)
      return clip(
        [
          asText(data.result),
          `${asText(data.iterations) || '1'} iteration(s)`,
          seconds !== null ? `${seconds.toFixed(1)}s` : '',
          `${asText(data.tokens) || '0'} tokens`,
        ]
          .filter(Boolean)
          .join(' - '),
      )
    }
    case 'DEEP_REPORT':
      return clip([asText(data.status), asText(data.error)].filter(Boolean).join(' - '))
    default: {
      try {
        return clip(JSON.stringify(data))
      } catch {
        return ''
      }
    }
  }
}

/* --- Task ----------------------------------------------------------------- */

export function taskModeLabel(task: TaskSummary | undefined | null): string {
  if (!task) return 'Task'
  if (task.is_hierarchical) return 'Hierarchical'
  return titleCase(task.root_mode || 'task')
}

export function taskLine(task: TaskSummary | undefined | null): string {
  if (!task) return 'No task recorded'
  const name = task.display_name || 'Task'
  const labels = task.class_labels ?? []
  if (labels.length >= 2) {
    const joined = labels.join(' vs ')
    return labels[0] === name ? joined : `${name}: ${joined}`
  }
  const outputs = task.regression_outputs ?? []
  if (outputs.length) return `${name}: ${outputs.join(', ')}`
  return name
}

/** Rebuild an editable task from the run summary so a re-run starts prefilled. */
export function taskSpecFromSummary(task: TaskSummary): TaskSpecInput {
  const byId = new Map(task.nodes.map((node) => [node.node_id, node]))
  const build = (nodeId: string, seen: Set<string>): TaskNodeInput | null => {
    const node = byId.get(nodeId)
    if (!node || seen.has(nodeId)) return null
    seen.add(nodeId)
    return {
      node_id: node.node_id,
      display_name: node.display_name,
      mode: node.mode,
      class_labels: [...node.class_labels],
      regression_outputs: [...node.regression_outputs],
      unit_by_output: {},
      required: node.required,
      children: node.child_ids
        .map((childId) => build(childId, seen))
        .filter((child): child is TaskNodeInput => child !== null),
    }
  }

  const rootId = task.nodes[0]?.node_id ?? 'root'
  const predictionType = task.is_hierarchical
    ? 'hierarchical'
    : task.root_mode === 'binary_classification'
      ? 'binary'
      : task.root_mode === 'multiclass_classification'
        ? 'multiclass'
        : task.root_mode === 'univariate_regression'
          ? 'regression_univariate'
          : 'regression_multivariate'

  return {
    prediction_type: predictionType,
    target_label: task.class_labels[0] ?? task.display_name ?? '',
    control_label: task.class_labels[1] ?? '',
    class_labels: [...task.class_labels],
    regression_outputs: [...task.regression_outputs],
    root: task.is_hierarchical ? build(rootId, new Set()) : null,
  }
}

/* --- Failure -------------------------------------------------------------- */

export interface FailureSummary {
  /** The exception class, when the worker reported one. */
  exception: string
  /** The exception message on its own, which is the part a reader can act on. */
  message: string
  /** One plain sentence naming what stopped and where. */
  lead: string
  stage: string | null
  step: RunStep | null
  /** A concrete next check for the failures that recur. Empty when unknown. */
  hint: string
}

const EXCEPTION_LINE = /^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit|Failure))\s*:\s*([\s\S]+)$/

/** Split "RuntimeError: something broke" into its class and its message. */
export function splitError(error: string | null | undefined): { exception: string; message: string } {
  const text = String(error ?? '').trim()
  if (!text) return { exception: '', message: '' }
  const match = EXCEPTION_LINE.exec(text)
  if (!match) return { exception: '', message: text }
  return { exception: match[1], message: match[2].trim() }
}

/**
 * Named checks for the failures that come back often enough to recognise. The
 * list stays short on purpose: a wrong guess costs more than no guess.
 */
const HINTS: { test: RegExp; hint: string }[] = [
  {
    test: /local backend|vllm|apptainer|singularity/i,
    hint: 'The self-hosted backend did not come up. Check the model name, the runtime, and that the machine has the GPUs the deployment asks for.',
  },
  {
    test: /api key|unauthorized|unauthorised|\b401\b|credential|forbidden|\b403\b/i,
    hint: 'The provider rejected the credential. Add or replace the API key in settings, then run again.',
  },
  { test: /rate limit|\b429\b|quota/i, hint: 'The provider throttled this run. Wait, or lower the number of parallel steps.' },
  {
    test: /timed out|timeout/i,
    hint: 'The provider did not answer inside the request timeout. Raise the timeout or choose a faster model.',
  },
  {
    test: /connection|unreachable|network|name resolution|dns|refused/i,
    hint: 'The provider could not be reached. Check the network and the base URL in settings.',
  },
  {
    test: /out of memory|cuda|oom/i,
    hint: 'The device ran out of memory. Lower the context length, the GPU memory fraction, or the parallel size.',
  },
  {
    test: /context length|maximum context|too many tokens/i,
    hint: 'The payload did not fit the model context. Raise the context window or lower the per-agent token budgets.',
  },
  {
    test: /no such file|filenotfound|not a directory/i,
    hint: 'A file the engine expected was not on disk. Re-check the participant directory and the data roots.',
  },
]

function hintFor(text: string): string {
  for (const entry of HINTS) if (entry.test.test(text)) return entry.hint
  return ''
}

/**
 * Turn a failed run into something readable.
 *
 * A stack trace answers "where in the code", which is the wrong first question.
 * The first question is what stopped and at which stage, so that is what this
 * derives; the trace stays available underneath.
 */
export function describeFailure(detail: RunDetail): FailureSummary {
  const state = detail.state
  const stages = state?.stages?.length ? state.stages : STAGE_NAMES
  const index = state?.current_stage ?? -1
  const stage = index >= 0 && index < stages.length ? stages[index] : null

  const steps = [...(state?.history ?? []), ...(state?.steps ?? [])]
  const step = [...steps].reverse().find((row) => row.status === 'failed') ?? null

  const { exception, message } = splitError(detail.error)
  const body = message || 'The worker exited without reporting a reason.'

  const lead = stage
    ? `The run stopped during ${stage.toLowerCase()}.`
    : detail.started_at
      ? 'The run stopped before it reached its first stage.'
      : 'The run never started.'

  return {
    exception,
    message: body,
    lead,
    stage,
    step,
    hint: hintFor(`${detail.error ?? ''} ${step?.error ?? ''}`),
  }
}

/* --- Structural audit ----------------------------------------------------- */

export interface AuditSection {
  name: string
  tokens: number
  featureKeys: number
}

export interface AuditChunk {
  index: number
  sections: string[]
  tokens: number
}

export interface AuditCoverage {
  all: number | null
  processed: number | null
  covered: number | null
  missing: number | null
  forcedRaw: number | null
  present: boolean
}

export interface AuditView {
  participantId: string
  targetCondition: string
  controlCondition: string
  taskMode: string
  nodeCount: number | null
  payloadTokens: number | null
  chunkBudget: number | null
  chunkCount: number | null
  inputMode: string
  coverage: AuditCoverage
  sections: AuditSection[]
  chunks: AuditChunk[]
  assertions: { key: string; ok: boolean }[]
  assertionsOk: boolean | null
  /** Whether anything at all was recovered, from either source. */
  present: boolean
  raw: Record<string, unknown>
}

/** The two spellings the engine has used for the same coverage tally. */
function coverageNumber(summary: Record<string, unknown>, ...keys: string[]): number | null {
  for (const key of keys) {
    const value = asNumber(summary[key])
    if (value !== null) return value
  }
  return null
}

/**
 * Read an audit from the run record.
 *
 * `audit_summary` is the current contract; an older record carries the same
 * fields inside the raw result, so both are merged with the summary winning.
 */
export function readAudit(detail: RunDetail): AuditView {
  const result = asRecord(detail.result)
  const summary = asRecord(detail.audit_summary)
  const raw: Record<string, unknown> = { ...result, ...summary }
  const coverageRaw = asRecord(raw.coverage_summary)

  const sections = asArray(raw.section_stats)
    .map((row) => asRecord(row))
    .map((row) => ({
      name: asText(row.name),
      tokens: asNumber(row.tokens) ?? 0,
      featureKeys: asNumber(row.feature_key_count) ?? 0,
    }))
    .filter((row) => row.name.length > 0)
    .sort((a, b) => b.tokens - a.tokens)

  const chunks = asArray(raw.chunk_stats)
    .map((row) => asRecord(row))
    .map((row, index) => ({
      index: asNumber(row.chunk_index) ?? index + 1,
      sections: asStringList(row.sections),
      tokens: asNumber(row.tokens) ?? 0,
    }))

  const assertionsRaw = asRecord(raw.assertions)
  const assertions = Object.entries(assertionsRaw)
    .filter(([, value]) => typeof value === 'boolean')
    .map(([key, value]) => ({ key, ok: value === true }))

  return {
    participantId: asText(raw.participant_id) || detail.participant_id,
    targetCondition: asText(raw.target_condition),
    controlCondition: asText(raw.control_condition),
    taskMode: asText(raw.prediction_task_root_mode),
    nodeCount: asNumber(raw.prediction_task_node_count),
    payloadTokens: asNumber(raw.predictor_payload_tokens),
    chunkBudget: asNumber(raw.chunk_budget_tokens),
    chunkCount: asNumber(raw.chunk_count),
    inputMode: asText(raw.predictor_input_mode),
    coverage: {
      all: coverageNumber(coverageRaw, 'all_count', 'all_feature_count'),
      processed: coverageNumber(coverageRaw, 'processed_count', 'processed_feature_count'),
      covered: coverageNumber(coverageRaw, 'covered_count', 'represented_feature_count'),
      missing: coverageNumber(coverageRaw, 'missing_count', 'missing_feature_count'),
      forcedRaw: coverageNumber(coverageRaw, 'forced_raw_count'),
      present: Object.keys(coverageRaw).length > 0,
    },
    sections,
    chunks,
    assertions,
    assertionsOk:
      typeof raw.assertions_ok === 'boolean'
        ? raw.assertions_ok
        : assertions.length > 0
          ? assertions.every((entry) => entry.ok)
          : null,
    present: Object.keys(raw).length > 0,
    raw,
  }
}

/** Assembler section ids read as words; a `#n` suffix is a split of one section. */
export function sectionLabel(name: string): { title: string; part: string } {
  const [base, part] = String(name ?? '').split('#')
  const title = titleCase(base.replace(/_raw$/, '')).replace(/\bRag\b/, 'RAG')
  return { title, part: part ? `part ${part}` : '' }
}

/** Assertion keys are schema names, so they read better spelled out. */
const ASSERTION_LABELS: Record<string, string> = {
  predictor_input_mode_present: 'The predictor input carries a mode',
  chunk_count_non_negative: 'The chunk count is well formed',
  coverage_summary_present: 'A coverage ledger was produced',
  task_mode_present: 'The task carries a prediction mode',
  invariant_ok: 'The coverage invariant held',
  missing_feature_count_zero: 'No feature was dropped',
  chunk_evidence_matches_count: 'Every chunk reported evidence',
  processed_raw_flag_consistent: 'The processed raw flag matches the payload',
}

export function assertionLabel(key: string): string {
  return ASSERTION_LABELS[key] ?? titleCase(key)
}
