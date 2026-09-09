/**
 * Derivations shared by the run list and the live console.
 *
 * The service sends loosely typed payloads (`state.prediction`, `event.data`)
 * because their shape depends on the task. Everything here reads them
 * defensively so a run that has emitted a single event still renders.
 */

import { titleCase } from '@/lib/format'
import type {
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
export function stageTimings(events: RunEvent[], stageCount: number, nowMs: number | null): StageTimings {
  const totals = new Array<number>(Math.max(0, stageCount)).fill(0)
  const visited = new Array<boolean>(Math.max(0, stageCount)).fill(false)
  let current = -1
  let since = 0

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
