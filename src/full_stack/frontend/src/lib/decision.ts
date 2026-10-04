/**
 * Structured decision models and evidence routing on the client.
 *
 * A decision model answers typed questions with calibrated probabilities and
 * writes no text, so it can only serve the Predictor. The service owns the
 * registry; this module mirrors its id pattern so a typed or saved id that the
 * catalog does not carry is still recognised before it is sent anywhere.
 */

import { useMemo } from 'react'
import { useCapabilities } from './hooks'
import type { DecisionConfig, DecisionModelSpec, ModelRow, OrchestrationMode } from './types'

export const DECISION_TOOLTIP =
  'Structured decision model: answers typed questions with calibrated probabilities. Predictor role only.'

/** Offered as the companion LLM when the current default cannot serve as one. */
export const COMPANION_FALLBACK_MODEL = 'deepseek/deepseek-v4-flash-0731'

/** The shipped values, used while a service that predates the section answers. */
export const DECISION_DEFAULTS: DecisionConfig = {
  provider: 'openrouter',
  choice_orders: 3,
  score_levels: 10,
  regression_refine: true,
  stability_threshold: 0.2,
  sufficiency_threshold: 0,
  compiler_model: '',
}

/**
 * Same rule as the service registry: any TypeSafe System One id ("jev-" and a
 * version) is a decision model, except the Jev Router, which routes text.
 */
const SYSTEM_ONE = /^(?:typesafe\/)?jev-(?!router)[a-z0-9.-]+$/

const normalise = (id: string) => {
  const text = String(id ?? '').trim().toLowerCase()
  return text.startsWith('~') ? text.slice(1) : text
}

export function isDecisionModelId(id: string | null | undefined, known?: ReadonlySet<string>): boolean {
  const key = normalise(id ?? '')
  if (!key) return false
  if (known?.has(key)) return true
  return SYSTEM_ONE.test(key)
}

/** A catalog row says what it is; only a row cached without `kind` falls back to the id. */
export function isDecisionRow(row: ModelRow | null | undefined, known?: ReadonlySet<string>): boolean {
  if (!row) return false
  if (row.kind) return row.kind === 'decision'
  return isDecisionModelId(row.id, known)
}

/** A catalog row for a registry entry the provider listing did not carry. */
export function rowFromSpec(spec: DecisionModelSpec): ModelRow {
  const [provider] = spec.model_id.split('/')
  return {
    id: spec.model_id,
    name: spec.label,
    provider: provider || 'typesafe',
    description: DECISION_TOOLTIP,
    context_length: spec.state_context_tokens,
    max_completion_tokens: null,
    prompt_usd_per_mtok: spec.input_price_per_million,
    completion_usd_per_mtok: spec.output_price_per_million ?? 0,
    is_free: false,
    modality: 'text->decision',
    input_modalities: ['text'],
    supports_structured_output: false,
    supports_tools: false,
    is_embedding: false,
    kind: 'decision',
    roles: spec.roles?.length ? spec.roles : ['predictor'],
  }
}

/** The decision models the service declares, keyed by normalised id. */
export function useDecisionModels(): { specs: DecisionModelSpec[]; ids: ReadonlySet<string> } {
  const capabilities = useCapabilities()
  return useMemo(() => {
    const specs = capabilities.data?.decision_models ?? []
    return { specs, ids: new Set(specs.map((spec) => normalise(spec.model_id))) }
  }, [capabilities.data])
}

/** A readable name for a decision model id, from the registry when it has one. */
export function decisionLabel(id: string, specs: DecisionModelSpec[]): string {
  const key = normalise(id)
  return specs.find((spec) => normalise(spec.model_id) === key)?.label ?? id
}

/* --- Orchestration routing ------------------------------------------------ */

export const DEFAULT_ORCHESTRATION_MODE: OrchestrationMode = 'auto'

export const ORCHESTRATION_CHOICES: { value: OrchestrationMode; label: string; hint: string }[] = [
  {
    value: 'auto',
    label: 'Auto',
    hint: 'Auto: run the multi-agent workflow only when the record does not fit the Predictor input',
  },
  { value: 'always', label: 'Always', hint: 'Always: also re-represent records that would fit' },
  { value: 'never', label: 'Never', hint: 'Never: always predict directly' },
]
