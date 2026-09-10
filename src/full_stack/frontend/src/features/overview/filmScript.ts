/**
 * The script behind the methodology film.
 *
 * Everything here is authored: the act order, the timings, the plan shape, and
 * the numbers on screen. Nothing is read from a participant and no run is made.
 * The names the film speaks (stages, agent roles, tool names, task shapes) are
 * resolved against `/capabilities` at render time, so the vocabulary follows
 * the engine even though the choreography does not.
 *
 * The film is a pure function of one clock value. Every position, opacity, and
 * counter below is derived from act progress, which is what makes scrubbing
 * land on a correct frame instead of a half-finished transition.
 */

import type { Capabilities } from '@/lib/types'

/* --- Timing helpers -------------------------------------------------------- */

export const clamp01 = (value: number): number => (value < 0 ? 0 : value > 1 ? 1 : value)

export const lerp = (from: number, to: number, u: number): number => from + (to - from) * u

export const easeOut = (u: number): number => 1 - Math.pow(1 - clamp01(u), 3)

export const easeInOut = (u: number): number => {
  const v = clamp01(u)
  return v < 0.5 ? 4 * v * v * v : 1 - Math.pow(-2 * v + 2, 3) / 2
}

/** Progress of the sub-window [from, to] at act progress `p`. */
export const seg = (p: number, from: number, to: number): number => clamp01((p - from) / (to - from))

/** Eased sub-window, the form used by nearly every entrance in the film. */
export const rise = (p: number, from: number, to: number): number => easeOut(seg(p, from, to))

/* --- Acts ------------------------------------------------------------------ */

export interface ActDefinition {
  key: string
  title: string
  /** Index into `capabilities.stages`, so the engine names the stage. */
  stage: number
  seconds: number
  prose: string
}

export const ACTS: ActDefinition[] = [
  {
    key: 'question',
    title: 'The question',
    stage: 0,
    seconds: 5.5,
    prose:
      'The task shape is chosen first, because it fixes the output contract: a binary comparison has to come back as one label, a probability, and the evidence behind both.',
  },
  {
    key: 'evidence',
    title: 'The evidence',
    stage: 0,
    seconds: 6,
    prose:
      'Everything measured for the participant is read into domain lanes, so the engine knows what it holds and, just as importantly, what is missing.',
  },
  {
    key: 'orchestration',
    title: 'Orchestration',
    stage: 1,
    seconds: 7.5,
    prose:
      'The Orchestrator turns the coverage map into a plan. Steps that depend on nothing share a column and will run together; a step that needs an earlier result sits to its right.',
  },
  {
    key: 'execution',
    title: 'Execution',
    stage: 2,
    seconds: 10,
    prose:
      'The Executor runs every unblocked step at once, holds the rest in a queue until a worker frees up, and repairs a failed step in place instead of discarding the plan.',
  },
  {
    key: 'integration',
    title: 'Integration',
    stage: 3,
    seconds: 6.5,
    prose:
      'The Integrator fuses the step outputs into one evidence bundle, tracks how much of the coverage map is accounted for, and splits any payload too large to pass in one piece.',
  },
  {
    key: 'prediction',
    title: 'Prediction',
    stage: 4,
    seconds: 6.5,
    prose:
      'The Predictor answers in the shape the task asked for, and every claim keeps a link back to the measurements that support it.',
  },
  {
    key: 'evaluation',
    title: 'Evaluation',
    stage: 5,
    seconds: 7,
    prose:
      'The Critic scores the answer against a checklist for this task shape. A weak first pass sends the run back to the Orchestrator for another iteration.',
  },
  {
    key: 'communication',
    title: 'Communication',
    stage: 6,
    seconds: 5.5,
    prose:
      'The Communicator writes the report from the evidence chain, and marks what was never measured as unknown rather than guessing at it.',
  },
]

export const ACT_STARTS: number[] = ACTS.reduce<number[]>((acc, _current, index) => {
  acc.push(index === 0 ? 0 : acc[index - 1] + ACTS[index - 1].seconds)
  return acc
}, [])

export const TOTAL_SECONDS = ACT_STARTS[ACT_STARTS.length - 1] + ACTS[ACTS.length - 1].seconds

export const actIndexAt = (time: number): number => {
  for (let index = ACTS.length - 1; index >= 0; index -= 1) {
    if (time >= ACT_STARTS[index]) return index
  }
  return 0
}

export const actProgressAt = (time: number, index: number): number =>
  clamp01((time - ACT_STARTS[index]) / ACTS[index].seconds)

/* --- Domains --------------------------------------------------------------- */

/**
 * Domain lanes. The keys are the engine's domain vocabulary; the counts are
 * illustrative and deliberately not taken from any scanned participant.
 */
export interface DomainLane {
  key: string
  label: string
  present: number
  total: number
}

export const LANES: DomainLane[] = [
  { key: 'DEMOGRAPHICS', label: 'Demographics', present: 8, total: 16 },
  { key: 'COGNITION', label: 'Cognition', present: 22, total: 100 },
  { key: 'BRAIN_MRI', label: 'Brain MRI', present: 31, total: 140 },
  { key: 'BIOLOGICAL_ASSAY', label: 'Biological assay', present: 46, total: 220 },
  { key: 'LIFESTYLE_ENVIRONMENT', label: 'Lifestyle and environment', present: 24, total: 80 },
]

export const LEAF_TOTAL = LANES.reduce((sum, lane) => sum + lane.present, 0)

/* --- The plan -------------------------------------------------------------- */

/**
 * One authored plan. `tool` names a tool the engine exposes; if a build renames
 * it, the resolver falls back to the tool at `fallback` so the film can never
 * show a tool that does not exist.
 *
 * The execution timings are authored rather than simulated, which keeps a
 * scrubbed frame identical to a played one.
 */
export interface PlanStep {
  id: string
  tool: string
  fallback: number
  subject: string
  wave: number
  deps: string[]
  /** Illustrative size of the step output, in tokens. */
  payload: number
  worker: number
  /** Act 4 sub-timeline, all in act progress. */
  queued: number
  start: number
  end: number
  /** The one step whose first attempt fails and is repaired in place. */
  failAt?: number
  retryAt?: number
  /** The one payload that overflows the budget and is split into two chunks. */
  chunked?: boolean
}

export const PLAN: PlanStep[] = [
  {
    id: 's1',
    tool: 'UnimodalCompressor',
    fallback: 0,
    subject: 'Demographics',
    wave: 0,
    deps: [],
    payload: 1400,
    worker: 0,
    queued: 0,
    start: 0.04,
    end: 0.22,
  },
  {
    id: 's2',
    tool: 'UnimodalCompressor',
    fallback: 0,
    subject: 'Cognition',
    wave: 0,
    deps: [],
    payload: 2100,
    worker: 1,
    queued: 0,
    start: 0.05,
    end: 0.26,
  },
  {
    id: 's3',
    tool: 'UnimodalCompressor',
    fallback: 0,
    subject: 'Brain MRI',
    wave: 0,
    deps: [],
    payload: 2600,
    worker: 2,
    queued: 0,
    start: 0.06,
    end: 0.31,
  },
  {
    id: 's4',
    tool: 'CodeExecutor',
    fallback: 3,
    subject: 'Assay panel',
    wave: 0,
    deps: [],
    payload: 1800,
    worker: 0,
    queued: 0,
    start: 0.24,
    end: 0.38,
  },
  {
    id: 's5',
    tool: 'FeatureSynthesizer',
    fallback: 4,
    subject: 'Derived indices',
    wave: 1,
    deps: ['s2', 's4'],
    payload: 2200,
    worker: 1,
    queued: 0.38,
    start: 0.4,
    end: 0.54,
  },
  {
    id: 's6',
    tool: 'AnomalyNarrativeBuilder',
    fallback: 6,
    subject: 'Strongest deviations',
    wave: 1,
    deps: ['s3'],
    payload: 3100,
    worker: 2,
    queued: 0.31,
    start: 0.33,
    end: 0.6,
    failAt: 0.43,
    retryAt: 0.47,
  },
  {
    id: 's7',
    tool: 'MultimodalNarrativeCreator',
    fallback: 1,
    subject: 'Cross-domain narrative',
    wave: 1,
    deps: ['s1', 's2', 's3'],
    payload: 9400,
    worker: 0,
    queued: 0.31,
    start: 0.4,
    end: 0.58,
    chunked: true,
  },
  {
    id: 's8',
    tool: 'ClinicalRelevanceRanker',
    fallback: 5,
    subject: 'Ranked findings',
    wave: 2,
    deps: ['s5', 's6'],
    payload: 1600,
    worker: 0,
    queued: 0.6,
    start: 0.62,
    end: 0.76,
  },
  {
    id: 's9',
    tool: 'HypothesisGenerator',
    fallback: 2,
    subject: 'Candidate mechanisms',
    wave: 2,
    deps: ['s6', 's7'],
    payload: 2400,
    worker: 2,
    queued: 0.6,
    start: 0.63,
    end: 0.78,
  },
  {
    id: 's10',
    tool: 'PhenotypeRepresentation',
    fallback: 7,
    subject: 'Structured phenotype',
    wave: 2,
    deps: ['s5', 's7'],
    payload: 2900,
    worker: 1,
    queued: 0.58,
    start: 0.6,
    end: 0.74,
  },
  {
    id: 's11',
    tool: 'DifferentialDiagnosis',
    fallback: 8,
    subject: 'Plausible alternatives',
    wave: 3,
    deps: ['s8', 's9'],
    payload: 2300,
    worker: 0,
    queued: 0.78,
    start: 0.8,
    end: 0.94,
  },
]

export const WAVE_COUNT = PLAN.reduce((max, step) => Math.max(max, step.wave + 1), 0)
export const WORKERS = 3
export const CHUNK_BUDGET = 8000
export const BUNDLE_TOKENS = PLAN.reduce((sum, step) => sum + step.payload, 0)

export const WAVE_NOTES: string[] = [
  'nothing to wait for',
  'waits on wave A',
  'waits on wave B',
  'waits on wave C',
]

/** Order the plan the way the executor releases it: by start, then by wave. */
export const RUN_ORDER: PlanStep[] = [...PLAN].sort((a, b) => a.start - b.start || a.wave - b.wave)

/* --- Later acts ------------------------------------------------------------ */

export interface EvidenceLink {
  lane: string
  leaf: string
  reading: string
  /** Illustrative contribution to the prediction, 0 to 1. */
  weight: number
}

export const EVIDENCE: EvidenceLink[] = [
  { lane: 'COGNITION', leaf: 'Working memory index', reading: '2.1 SD below reference', weight: 0.82 },
  { lane: 'BRAIN_MRI', leaf: 'Hippocampal volume', reading: '1.6 SD below reference', weight: 0.64 },
  { lane: 'BIOLOGICAL_ASSAY', leaf: 'Inflammatory marker', reading: '3.4 mg/L, elevated', weight: 0.41 },
]

export const PREDICTION = {
  probability: 0.78,
  confidence: 0.71,
  supporting: 3,
  contradicting: 1,
}

export interface CheckItem {
  label: string
  /** Outcome on the first pass and on the second. */
  first: 'pass' | 'fail'
  second: 'pass'
}

export const CHECKLIST: CheckItem[] = [
  { label: 'Output matches the task shape', first: 'pass', second: 'pass' },
  { label: 'Every claim cites a measurement', first: 'fail', second: 'pass' },
  { label: 'Alternatives were considered', first: 'pass', second: 'pass' },
  { label: 'Uncertainty is stated', first: 'fail', second: 'pass' },
  { label: 'No inference beyond the evidence', first: 'pass', second: 'pass' },
]

export const REPORT_SECTIONS: { title: string; lines: number }[] = [
  { title: 'Summary', lines: 2 },
  { title: 'Evidence by domain', lines: 3 },
  { title: 'Prediction and confidence', lines: 2 },
  { title: 'Limitations', lines: 1 },
]

export const UNKNOWNS: string[] = ['Sleep actigraphy: not measured', 'Family history: not recorded']

/* --- Vocabulary resolved from the engine ----------------------------------- */

export interface FilmAgent {
  label: string
  summary: string
}

export interface ResolvedStep extends PlanStep {
  toolName: string
  family: string
  familyIndex: number
}

export interface FilmShape {
  value: string
  label: string
  summary: string
  needs: string[]
}

export interface FilmVocabulary {
  stages: string[]
  agents: Record<string, FilmAgent>
  steps: ResolvedStep[]
  shapes: FilmShape[]
  chosenShape: number
  familyCount: number
}

const FALLBACK_AGENT: Record<string, FilmAgent> = {
  orchestrator: { label: 'Orchestrator', summary: 'Writes the execution plan.' },
  executor: { label: 'Executor', summary: 'Runs the plan steps.' },
  integrator: { label: 'Integrator', summary: 'Fuses the step outputs.' },
  predictor: { label: 'Predictor', summary: 'Produces the phenotype outputs.' },
  critic: { label: 'Critic', summary: 'Scores the prediction.' },
  communicator: { label: 'Communicator', summary: 'Writes the report.' },
}

/**
 * Bind the authored script to the engine's own words. Tool names resolve by
 * name, then by position, so a renamed tool degrades to a real name rather than
 * to a name this file invented.
 */
export function buildVocabulary(capabilities: Capabilities): FilmVocabulary {
  const tools = capabilities.tools ?? []
  const families = [...new Set(tools.map((tool) => tool.family || 'other'))]

  const steps: ResolvedStep[] = PLAN.map((step) => {
    const match = tools.find((tool) => tool.name === step.tool) ?? tools[step.fallback % Math.max(1, tools.length)]
    const family = match?.family || 'other'
    return {
      ...step,
      toolName: match?.name ?? step.tool,
      family,
      familyIndex: Math.max(0, families.indexOf(family)) % 8,
    }
  })

  const agents: Record<string, FilmAgent> = { ...FALLBACK_AGENT }
  for (const agent of capabilities.agents ?? []) {
    agents[agent.role] = { label: agent.label, summary: agent.summary }
  }

  const shapes: FilmShape[] = (capabilities.prediction_types ?? []).map((type) => ({
    value: type.value,
    label: type.label,
    summary: type.summary,
    needs: type.needs ?? [],
  }))

  const binary = shapes.findIndex((shape) => shape.value === 'binary')

  return {
    stages: capabilities.stages ?? [],
    agents,
    steps,
    shapes,
    chosenShape: binary >= 0 ? binary : 0,
    familyCount: Math.max(1, families.length),
  }
}

/* --- Stage geometry -------------------------------------------------------- */

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

export interface StageLayout {
  width: number
  height: number
  wide: boolean
  hud: Box
  scene: Box
  /** Scene minus the bottom strip, used by the acts that carry a strip. */
  work: Box
  strip: Box
}

/**
 * One SVG unit is one CSS pixel, so type never scales below its authored size.
 * The layout reflows instead of shrinking, which is what keeps the stage
 * legible from a narrow column up to the full content width.
 */
export function layoutFor(width: number): StageLayout {
  const wide = width >= 900
  const pad = wide ? 22 : 14
  const height = wide ? 540 : 820
  const hud: Box = { x: pad, y: 14, w: width - pad * 2, h: 44 }
  const sceneY = hud.y + hud.h + (wide ? 18 : 14)
  const scene: Box = { x: pad, y: sceneY, w: width - pad * 2, h: height - sceneY - pad }
  // The footer band carries a few figures, not a panel of them, so it gives
  // height back to the diagram above it. It cannot give back everything: the
  // execution act draws its worker pool down here and needs about 114px, while
  // orchestration needs 314 above once the agent plate sits over the graph.
  // This is the value that satisfies both without either one clipping.
  const stripH = wide ? 104 : 96
  const strip: Box = { x: scene.x, y: scene.y + scene.h - stripH, w: scene.w, h: stripH }
  const work: Box = { x: scene.x, y: scene.y, w: scene.w, h: scene.h - stripH - 14 }
  return { width, height, wide, hud, scene, work, strip }
}

/** Rough text metrics, good enough to keep a label inside its own box. */
export function fit(text: string, maxWidth: number, fontSize: number): string {
  // Measured against the rendered face rather than guessed: the tool names are
  // CamelCase, and at 0.54 the estimate was optimistic enough that a name which
  // did fit was still being cut.
  const perChar = fontSize * 0.575
  const room = Math.floor(maxWidth / perChar)
  if (text.length <= room) return text
  if (room <= 1) return ''
  return `${text.slice(0, Math.max(1, room - 1)).trimEnd()}…`
}
