/**
 * Readers for the run artifacts.
 *
 * The engine writes these files across several schema generations, and a run
 * that ended early legitimately leaves whole sections null. Every accessor here
 * therefore returns a defined value and the screens decide how to say that
 * something is missing, rather than guarding at each use site.
 */

import type { ReportBundle } from '@/lib/types'

export type Dict = Record<string, unknown>

export function asDict(value: unknown): Dict {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Dict) : {}
}

export function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function asText(value: unknown, fallback = ''): string {
  if (value === null || value === undefined) return fallback
  if (typeof value === 'string') return value.trim() || fallback
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return fallback
}

export function asNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function asBoolean(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  return null
}

export function asStrings(value: unknown): string[] {
  return asList(value)
    .map((entry) => asText(entry))
    .filter((entry) => entry.length > 0)
}

/* --- Prediction tree ------------------------------------------------------ */

export interface Finding {
  domain: string
  finding: string
  direction: string
  z: number | null
  relevance: string
}

export interface PredictionNode {
  nodeId: string
  path: string
  mode: string
  displayName: string
  predictedLabel: string
  probabilities: { label: string; value: number }[]
  values: { output: string; value: number | null; unit: string }[]
  confidenceLevel: string
  confidenceScore: number | null
  keyFindings: Finding[]
  reasoning: string[]
  evidenceFor: string[]
  evidenceAgainst: string[]
  uncertainty: string[]
  children: PredictionNode[]
}

const Z_IN_TEXT = /\bz\s*=\s*(-?\d+(?:\.\d+)?)/i

function findingZ(entry: Dict): number | null {
  for (const key of ['z_score', 'z', 'zscore']) {
    const value = asNumber(entry[key])
    if (value !== null) return value
  }
  const match = Z_IN_TEXT.exec(asText(entry.finding))
  return match ? asNumber(match[1]) : null
}

function readFindings(value: unknown): Finding[] {
  return asList(value)
    .map((entry) => asDict(entry))
    .filter((entry) => Object.keys(entry).length > 0)
    .map((entry) => ({
      domain: asText(entry.domain, 'Unspecified domain'),
      finding: asText(entry.finding, asText(entry.description)),
      direction: asText(entry.direction).toUpperCase(),
      z: findingZ(entry),
      relevance: asText(entry.relevance_to_prediction ?? entry.relevance),
    }))
    .filter((entry) => entry.finding.length > 0)
}

/** Index the task specification tree by node id so units and names are reachable. */
function indexSpec(root: Dict, into: Map<string, Dict> = new Map()): Map<string, Dict> {
  const id = asText(root.node_id)
  if (id) into.set(id, root)
  for (const child of asList(root.children)) indexSpec(asDict(child), into)
  return into
}

function readNode(raw: Dict, spec: Map<string, Dict>): PredictionNode {
  const nodeId = asText(raw.node_id, 'root')
  const specNode = spec.get(nodeId) ?? {}
  const units = asDict(specNode.unit_by_output)

  const classification = asDict(raw.classification)
  const probabilities = Object.entries(asDict(classification.probabilities))
    .map(([label, value]) => ({ label, value: asNumber(value) ?? 0 }))
    .sort((a, b) => b.value - a.value)

  const regression = asDict(raw.regression)
  const rawValues = asDict(regression.values)
  const declaredOutputs = asStrings(specNode.regression_outputs)
  const outputNames = declaredOutputs.length
    ? Array.from(new Set([...declaredOutputs, ...Object.keys(rawValues)]))
    : Object.keys(rawValues)
  const values = outputNames.map((output) => ({
    output,
    value: asNumber(rawValues[output]),
    unit: asText(units[output]),
  }))

  return {
    nodeId,
    path: asText(raw.path, nodeId),
    mode: asText(raw.mode, asText(specNode.mode, 'unknown')),
    displayName: asText(specNode.display_name, nodeId),
    predictedLabel: asText(classification.predicted_label),
    probabilities,
    values,
    confidenceLevel: asText(raw.confidence_level),
    confidenceScore: asNumber(raw.confidence_score),
    keyFindings: readFindings(raw.key_findings),
    reasoning: asStrings(raw.reasoning_chain),
    evidenceFor: asStrings(raw.supporting_evidence_for),
    evidenceAgainst: asStrings(raw.supporting_evidence_against),
    uncertainty: asStrings(raw.uncertainty_factors),
    children: asList(raw.children).map((child) => readNode(asDict(child), spec)),
  }
}

export function flattenNodes(node: PredictionNode | null): PredictionNode[] {
  if (!node) return []
  return [node, ...node.children.flatMap((child) => flattenNodes(child))]
}

export function isClassificationMode(mode: string): boolean {
  return mode.includes('classification')
}

export function isRegressionMode(mode: string): boolean {
  return mode.includes('regression')
}

/* --- Assembled view of one bundle ----------------------------------------- */

export interface Evaluation {
  verdict: string
  compositeScore: number | null
  confidenceInVerdict: number | null
  scoreBreakdown: { key: string; value: number }[]
  checklist: { key: string; passed: boolean; active: boolean }[]
  checklistPassed: number | null
  checklistTotal: number | null
  strengths: string[]
  weaknesses: string[]
  suggestions: { issue: string; suggestion: string; priority: string }[]
  domainsMissed: string[]
  reasoning: string
  summary: string
  source: 'critic' | 'report' | 'none'
}

export interface Coverage {
  allFeatures: number | null
  processedFeatures: number | null
  representedFeatures: number | null
  missingCount: number | null
  missingFeatures: string[]
  invariantOk: boolean | null
  present: boolean
}

export interface TokenLine {
  component: string
  calls: number
  prompt: number
  completion: number
  total: number
}

export interface ReportView {
  participantId: string
  outputDir: string
  exists: boolean
  hasPerformance: boolean
  targetCondition: string
  controlCondition: string
  taskRoot: Dict
  taskMode: string
  taskName: string
  classLabels: string[]
  regressionOutputs: string[]
  unitByOutput: Record<string, string>
  timestamp: string
  durationSeconds: number | null
  iterations: number | null
  selectedIteration: number | null
  selectionReason: string
  primaryOutput: string
  confidenceLevel: string
  rootConfidence: number | null
  verdict: string
  totalTokens: number | null
  promptTokens: number | null
  completionTokens: number | null
  root: PredictionNode | null
  isHierarchical: boolean
  findings: Finding[]
  findingsSource: 'prediction' | 'report' | 'none'
  reasoning: string[]
  evidenceFor: string[]
  evidenceAgainst: string[]
  uncertainty: string[]
  clinicalSummary: string
  coverage: Coverage
  domainCoverage: { domain: string; present: number; total: number; percentage: number | null; missing: number; tokens: number | null; available: boolean }[]
  evaluation: Evaluation
  tokenLines: TokenLine[]
  plan: { id: string; steps: number | null; domains: string[] }
  dataflow: Dict
}

const CHECK_KEYS = [
  'has_required_outputs',
  'output_schema_valid',
  'classification_probabilities_valid',
  'regression_values_valid',
  'hierarchy_consistent',
  'sufficient_coverage',
  'evidence_based_reasoning',
  'clinically_relevant',
  'logically_coherent',
  'critical_domains_processed',
]

function readEvaluation(performance: Dict, patient: Dict): Evaluation {
  const critic = asDict(performance.critic_evaluation ?? performance.evaluation)
  const fallback = asDict(patient.evaluation)
  const source: Evaluation['source'] = Object.keys(critic).length
    ? 'critic'
    : Object.keys(fallback).length
      ? 'report'
      : 'none'
  const primary = source === 'critic' ? critic : fallback

  const checklistRaw = asDict(critic.checklist)
  const active = asStrings(checklistRaw.active_checks)
  const keys = Array.from(new Set([...active, ...CHECK_KEYS])).filter((key) => key in checklistRaw || active.includes(key))
  const checklist = keys.map((key) => ({
    key,
    passed: asBoolean(checklistRaw[key]) ?? false,
    active: active.length === 0 || active.includes(key),
  }))

  const activeChecks = checklist.filter((entry) => entry.active)
  const checklistPassed =
    activeChecks.length > 0 ? activeChecks.filter((entry) => entry.passed).length : asNumber(fallback.checklist_passed)
  const checklistTotal = activeChecks.length > 0 ? activeChecks.length : asNumber(fallback.checklist_total)

  return {
    verdict: asText(primary.verdict, asText(performance.critic_verdict)).toUpperCase(),
    compositeScore: asNumber(primary.composite_score),
    confidenceInVerdict: asNumber(primary.confidence_in_verdict),
    scoreBreakdown: Object.entries(asDict(critic.score_breakdown)).map(([key, value]) => ({
      key,
      value: asNumber(value) ?? 0,
    })),
    checklist,
    checklistPassed,
    checklistTotal,
    strengths: asStrings(critic.strengths),
    weaknesses: asStrings(critic.weaknesses),
    suggestions: asList(critic.improvement_suggestions)
      .map((entry) => asDict(entry))
      .map((entry) => ({
        issue: asText(entry.issue),
        suggestion: asText(entry.suggestion),
        priority: asText(entry.priority).toUpperCase(),
      }))
      .filter((entry) => entry.issue || entry.suggestion),
    domainsMissed: asStrings(critic.domains_missed),
    reasoning: asText(critic.reasoning),
    summary: asText(critic.concise_summary),
    source,
  }
}

function readCoverage(performance: Dict, patient: Dict): Coverage {
  const raw = Object.keys(asDict(performance.coverage_summary)).length
    ? asDict(performance.coverage_summary)
    : asDict(asDict(patient.execution).coverage_summary)
  return {
    allFeatures: asNumber(raw.all_feature_count),
    processedFeatures: asNumber(raw.processed_feature_count),
    representedFeatures: asNumber(raw.represented_feature_count),
    missingCount: asNumber(raw.missing_feature_count),
    missingFeatures: asStrings(raw.missing_features),
    invariantOk: asBoolean(raw.invariant_ok),
    present: Object.keys(raw).length > 0,
  }
}

export function groupTokenCalls(calls: unknown): TokenLine[] {
  const grouped = new Map<string, TokenLine>()
  for (const entry of asList(calls)) {
    const call = asDict(entry)
    const component = asText(call.component, 'unknown')
    const prompt = asNumber(call.prompt_tokens) ?? 0
    const completion = asNumber(call.completion_tokens) ?? 0
    const total = asNumber(call.total) ?? asNumber(call.total_tokens) ?? prompt + completion
    const line = grouped.get(component) ?? { component, calls: 0, prompt: 0, completion: 0, total: 0 }
    line.calls += 1
    line.prompt += prompt
    line.completion += completion
    line.total += total
    grouped.set(component, line)
  }
  return [...grouped.values()].sort((a, b) => b.total - a.total)
}

export function buildView(bundle: ReportBundle): ReportView {
  const performance = asDict(bundle.performance_report)
  const patient = asDict(bundle.patient_report)
  const patientPrediction = asDict(patient.prediction)
  const overview = asDict(bundle.data_overview)

  const spec = Object.keys(asDict(performance.prediction_task_spec)).length
    ? asDict(performance.prediction_task_spec)
    : asDict(patientPrediction.prediction_task_spec)
  const taskRoot = asDict(spec.root)
  const specIndex = indexSpec(taskRoot)

  const predictionResult = asDict(performance.prediction_result)
  const rawRoot = Object.keys(asDict(predictionResult.root_prediction)).length
    ? asDict(predictionResult.root_prediction)
    : asDict(patientPrediction.root_prediction)
  const root = Object.keys(rawRoot).length ? readNode(rawRoot, specIndex) : null
  const nodes = flattenNodes(root)

  const nodeFindings = nodes.flatMap((node) => node.keyFindings)
  const reportFindings = readFindings(patient.key_findings)
  const findings = nodeFindings.length ? nodeFindings : reportFindings
  const findingsSource: ReportView['findingsSource'] = nodeFindings.length
    ? 'prediction'
    : reportFindings.length
      ? 'report'
      : 'none'

  const nodeReasoning = nodes.flatMap((node) => node.reasoning)
  const usage = asDict(performance.token_usage)
  const plan = asDict(performance.plan_summary)

  const primaryOutput =
    asText(patientPrediction.primary_output) ||
    asText(predictionResult.classification) ||
    (root ? nodeOutputText(root) : '')

  return {
    participantId: asText(performance.participant_id, bundle.participant_id || 'Unknown participant'),
    outputDir: asText(bundle.output_dir),
    exists: Boolean(bundle.exists),
    hasPerformance: Object.keys(performance).length > 0,
    targetCondition: asText(performance.target_condition, asText(patientPrediction.target_condition)),
    controlCondition: asText(performance.control_condition, asText(patientPrediction.control_condition)),
    taskRoot,
    taskMode: asText(taskRoot.mode, asText(patientPrediction.root_mode)),
    taskName: asText(taskRoot.display_name),
    classLabels: asStrings(taskRoot.class_labels),
    regressionOutputs: asStrings(taskRoot.regression_outputs),
    unitByOutput: Object.fromEntries(
      Object.entries(asDict(taskRoot.unit_by_output)).map(([key, value]) => [key, asText(value)]),
    ),
    timestamp: asText(performance.execution_timestamp, asText(patient.generated_at)),
    durationSeconds: asNumber(performance.total_duration_seconds),
    iterations: asNumber(performance.iterations) ?? asNumber(asDict(patient.execution).iterations),
    selectedIteration:
      asNumber(performance.selected_iteration) ?? asNumber(asDict(patient.execution).selected_iteration),
    selectionReason:
      asText(performance.selection_reason) || asText(asDict(patient.execution).selection_reason),
    primaryOutput,
    confidenceLevel: asText(predictionResult.confidence, root?.confidenceLevel ?? ''),
    rootConfidence: asNumber(predictionResult.root_confidence) ?? root?.confidenceScore ?? null,
    verdict: asText(performance.critic_verdict, asText(asDict(patient.evaluation).verdict)).toUpperCase(),
    totalTokens: asNumber(usage.total_tokens),
    promptTokens: asNumber(usage.prompt_tokens),
    completionTokens: asNumber(usage.completion_tokens),
    root,
    isHierarchical: Boolean(root && root.children.length > 0),
    findings,
    findingsSource,
    reasoning: nodeReasoning.length ? nodeReasoning : asStrings(patient.reasoning),
    evidenceFor: nodes.flatMap((node) => node.evidenceFor),
    evidenceAgainst: nodes.flatMap((node) => node.evidenceAgainst),
    uncertainty: nodes.flatMap((node) => node.uncertainty),
    clinicalSummary: asText(patient.clinical_summary),
    coverage: readCoverage(performance, patient),
    domainCoverage: Object.entries(asDict(overview.domain_coverage)).map(([domain, value]) => {
      const entry = asDict(value)
      return {
        domain,
        present: asNumber(entry.present_leaves) ?? 0,
        total: asNumber(entry.total_leaves) ?? 0,
        percentage: asNumber(entry.coverage_percentage),
        missing: asNumber(entry.missing_count) ?? 0,
        tokens: asNumber(entry.total_tokens),
        available: asBoolean(entry.is_available) ?? false,
      }
    }),
    evaluation: readEvaluation(performance, patient),
    tokenLines: groupTokenCalls(usage.calls),
    plan: {
      id: asText(plan.plan_id),
      steps: asNumber(plan.total_steps),
      domains: asStrings(plan.priority_domains),
    },
    dataflow: asDict(performance.dataflow_summary),
  }
}

/** One line describing what a node predicted, whatever its mode. */
export function nodeOutputText(node: PredictionNode): string {
  if (node.predictedLabel) return node.predictedLabel
  if (node.values.length) {
    return node.values
      .map((entry) => `${entry.output}: ${entry.value === null ? 'n/a' : entry.value}`)
      .join('; ')
  }
  return 'Not recorded'
}

export const VERDICT_TONE: Record<string, 'positive' | 'critical' | 'neutral'> = {
  SATISFACTORY: 'positive',
  UNSATISFACTORY: 'critical',
}

export function directionTone(direction: string): 'critical' | 'info' | 'positive' | 'neutral' {
  if (direction.includes('HIGH')) return 'critical'
  if (direction.includes('LOW')) return 'info'
  if (direction.includes('NORMAL')) return 'positive'
  return 'neutral'
}
