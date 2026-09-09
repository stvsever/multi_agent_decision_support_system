/** Types mirroring the dashboard service contract. */

export type Theme = 'system' | 'light' | 'dark'
export type Accent = 'indigo' | 'violet' | 'teal' | 'amber' | 'rose' | 'slate'
export type Density = 'comfortable' | 'compact'
export type BackendName = 'openrouter' | 'openai' | 'local'
export type ReasoningEffort = 'provider_default' | 'off' | 'low' | 'medium' | 'high'
export type PredictionType =
  | 'binary'
  | 'multiclass'
  | 'regression_univariate'
  | 'regression_multivariate'
  | 'hierarchical'
export type NodeMode =
  | 'binary_classification'
  | 'multiclass_classification'
  | 'univariate_regression'
  | 'multivariate_regression'
export type AgentRole = 'orchestrator' | 'integrator' | 'predictor' | 'critic' | 'communicator' | 'tool'
export type RunStatus = 'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled'

export const AGENT_ROLES: AgentRole[] = [
  'orchestrator',
  'integrator',
  'predictor',
  'critic',
  'communicator',
  'tool',
]

export const INSTRUCTION_SLOTS = [
  'global',
  'orchestrator',
  'executor',
  'tools',
  'integrator',
  'predictor',
  'critic',
  'communicator',
] as const
export type InstructionSlot = (typeof INSTRUCTION_SLOTS)[number]

export type RoleMap<T> = Record<AgentRole, T>

export interface DashboardConfig {
  version: number
  connection: {
    backend: BackendName
    openrouter_base_url: string
    openrouter_site_url: string
    openrouter_app_name: string
    request_timeout_seconds: number
    max_retries: number
    retry_delay_seconds: number
  }
  models: {
    default_model: string
    reasoning_effort: ReasoningEffort
    context_window: number
    embedding_model: string
    role_models: RoleMap<string>
    role_max_tokens: RoleMap<number>
    role_temperatures: RoleMap<number>
  }
  engine: {
    max_iterations: number
    executor_max_workers: number
    auto_repair_enabled: boolean
    detailed_tool_logging: boolean
    verbose: boolean
  }
  token_budget: {
    total_budget: number
    max_agent_input_tokens: number
    max_agent_output_tokens: number
    max_tool_input_tokens: number
    max_tool_output_tokens: number
    orchestrator_budget: number
    executor_budget_per_step: number
    fusion_budget: number
    integrator_budget: number
    predictor_budget: number
    critic_budget: number
    communicator_budget: number
  }
  local: {
    model_name: string
    max_tokens: number
    engine: 'auto' | 'vllm' | 'transformers'
    dtype: string
    quantization: string
    kv_cache_dtype: string
    attn_implementation: string
    tensor_parallel_size: number
    pipeline_parallel_size: number
    gpu_memory_utilization: number
    max_model_len: number
    enforce_eager: boolean
    trust_remote_code: boolean
  }
  batch: { concurrency: number; continue_on_error: boolean; run_timeout_seconds: number }
  cost: { warn_above_usd: number; block_above_usd: number; currency_decimals: number }
  workspace: { data_roots: string[]; output_dir: string; auto_open_report: boolean }
  appearance: {
    theme: Theme
    accent: Accent
    density: Density
    font_scale: number
    reduced_motion: boolean
    show_advanced_by_default: boolean
    flow_animate_edges: boolean
    flow_direction: 'LR' | 'TB'
    flow_edge_style: 'bezier' | 'smoothstep' | 'straight'
    flow_show_tokens: boolean
    numeric_locale: string
  }
  instructions: Partial<Record<InstructionSlot, string>>
  onboarding_complete: boolean
  tour_completed: string[]
}

export interface CredentialStatus {
  provider: string
  configured: boolean
  masked: string
  source: 'stored' | 'environment' | 'none'
}

export interface SettingsResponse {
  config: DashboardConfig
  effective: Record<string, unknown>
  credentials: Record<string, CredentialStatus>
}

export interface AccountStatus {
  configured: boolean
  valid: boolean
  reason?: string
  label?: string
  usage_usd?: number | null
  limit_usd?: number | null
  remaining_usd?: number | null
  is_free_tier?: boolean
  catalog_ready?: boolean
  catalog_error?: string
}

export interface ModelRow {
  id: string
  name: string
  provider: string
  description: string
  context_length: number | null
  max_completion_tokens: number | null
  prompt_usd_per_mtok: number | null
  completion_usd_per_mtok: number | null
  is_free: boolean
  modality: string
  input_modalities: string[]
  supports_structured_output: boolean
  supports_tools: boolean
  is_embedding: boolean
  created?: number
}

export interface CatalogResponse {
  models: ModelRow[]
  total: number
  providers: string[]
  fetched_at: number
  stale: boolean
  error?: string
}

export interface FileCheck {
  file: string
  key: string
  present: boolean
  valid: boolean
  size?: number
  issue: string
}

export interface DomainCoverage {
  present_leaves: number
  total_leaves: number
  coverage_percentage: number
  missing_count: number
  total_tokens: number
  is_available: boolean
}

export interface Participant {
  id: string
  directory: string
  name: string
  valid: boolean
  files: FileCheck[]
  missing: string[]
  input_tokens: number
  domains: string[]
  domain_coverage: Record<string, DomainCoverage>
  token_budget?: number
}

export interface ParticipantsResponse {
  participants: Participant[]
  roots: { root: string; label: string; found: number; bundled: boolean }[]
  count: number
}

export interface TaskNodeInput {
  node_id: string
  display_name: string
  mode: NodeMode
  class_labels: string[]
  regression_outputs: string[]
  unit_by_output: Record<string, string>
  required: boolean
  children: TaskNodeInput[]
}

export interface TaskSpecInput {
  prediction_type: PredictionType
  target_label: string
  control_label: string
  class_labels: string[]
  regression_outputs: string[]
  root?: TaskNodeInput | null
}

export interface RunOverrides {
  connection?: Record<string, unknown>
  models?: Record<string, unknown>
  engine?: Record<string, unknown>
  token_budget?: Record<string, unknown>
  local?: Record<string, unknown>
  instructions?: Record<string, string>
}

export interface EstimateLine {
  role: string
  model: string
  calls: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  usd: number | null
  price_known: boolean
  prompt_usd_per_mtok: number | null
  completion_usd_per_mtok: number | null
}

export interface Estimate {
  input_tokens: number
  iterations: number
  plan_steps: number
  include_deep_report: boolean
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  usd: number | null
  usd_low: number | null
  usd_high: number | null
  spread: number
  fully_priced: boolean
  lines: EstimateLine[]
  budget_utilisation: number | null
}

export interface EstimateResponse {
  participants: (Estimate & { participant_dir: string; id: string })[]
  totals: {
    count: number
    total_tokens: number
    usd: number | null
    usd_low: number | null
    usd_high: number | null
    fully_priced: boolean
  }
  guards: { warn_above_usd: number; block_above_usd: number; warns: boolean; blocks: boolean }
  effective_models: Record<string, string>
}

export interface CostLine {
  model: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  usd: number | null
  prompt_usd_per_mtok: number | null
  completion_usd_per_mtok: number | null
}

export interface CostSummary {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens: number
  usd: number | null
  fully_priced?: boolean
  lines: CostLine[]
}

export interface RunStep {
  id: number
  tool: string
  desc: string
  status: 'running' | 'complete' | 'failed' | 'repairing'
  tokens: number
  duration?: number
  preview?: string
  error?: string
  iteration?: number
}

export interface RunState {
  participant_id?: string
  target?: string
  control?: string | null
  status: string
  current_stage: number
  stages: string[]
  steps: RunStep[]
  history: RunStep[]
  iteration: number
  max_iterations?: number
  total_tokens: number
  progress: number
  max_steps: number
  completed: boolean
  prediction?: Record<string, unknown> | null
  critic?: Record<string, unknown> | null
  critic_summary?: string | null
  completion?: Record<string, unknown> | null
  fusion_data?: Record<string, unknown>
  plans?: Record<string, unknown>
}

export interface RunEvent {
  id: number
  time: string
  ts?: string
  type: string
  data: Record<string, unknown>
}

export interface GraphNode {
  id: string
  type: 'agent' | 'tool'
  label: string
  role?: string
  family?: string
  step_id?: number
  stage: number
  rank: number
  lane: number
  detail: string
  reasoning?: string
  status?: string
  meta: Record<string, unknown>
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  kind: 'sequential' | 'fan_out' | 'fan_in' | 'mesh' | 'feedback'
  label?: string
  dispatch?: boolean
}

export interface RunGraph {
  plan_id: string
  iteration: number
  nodes: GraphNode[]
  edges: GraphEdge[]
  lanes: { rank: number; size: number; parallel: boolean }[]
  meta: {
    step_count: number
    depth: number
    max_parallel: number
    priority_domains: string[]
    explanation: string
    reasoning: string
    fusion_strategy: string
    estimated_tokens: number
  }
}

export interface TaskSummary {
  root_mode: NodeMode
  display_name: string
  node_count: number
  is_hierarchical: boolean
  class_labels: string[]
  regression_outputs: string[]
  nodes: {
    node_id: string
    display_name: string
    mode: NodeMode
    class_labels: string[]
    regression_outputs: string[]
    required: boolean
    child_ids: string[]
  }[]
}

export interface RunSummary {
  id: string
  label: string
  participant_id: string
  participant_dir: string
  status: RunStatus
  audit: boolean
  batch_id: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
  task: TaskSummary
  estimate: { usd: number | null; total_tokens: number }
  cost: { usd: number | null; total_tokens: number }
  stage: number
  progress: number
  max_steps: number
  verdict?: string
  prediction?: string
}

export interface RunDetail extends RunSummary {
  state: RunState
  events: RunEvent[]
  usage: { by_model: Record<string, { prompt: number; completion: number; calls: number }>; totals: { prompt: number; completion: number; calls: number } }
  cost: CostSummary & { usd: number | null; total_tokens: number }
  estimate: Estimate
  result: Record<string, unknown> | null
  graph: RunGraph | null
  logs: string[]
  traceback: string | null
  effective_config: DashboardConfig
}

export interface BatchStatus {
  id: string
  label: string
  created_at: string
  finished_at?: string
  run_ids: string[]
  concurrency: number
  continue_on_error: boolean
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  runs: RunSummary[]
  completed: number
  total: number
  succeeded: number
  failed: number
  spend_usd: number
}

export type Band =
  | 'very_high'
  | 'high'
  | 'high_normal'
  | 'normal'
  | 'low_normal'
  | 'low'
  | 'very_low'
  | 'missing'

export interface OntologyFeature {
  feature: string
  value: unknown
  z_score: number | null
  band: Band
  ref_range?: string | null
  significance?: string | null
  percentile?: string | null
}

export interface OntologyNode {
  id: string
  label: string
  path: string[]
  depth: number
  kind: 'domain' | 'node' | 'leaf'
  score: number | null
  band: Band
  mean_abs_score: number | null
  leaf_count: number
  present_leaves: number
  feature_count: number
  features: OntologyFeature[]
  children: OntologyNode[]
}

export interface Ontology {
  participant_id: string
  domains: OntologyNode[]
  extremes: { path: string[]; label: string; score: number; band: Band }[]
  summary: {
    domain_count: number
    node_count: number
    leaf_count: number
    present_leaves: number
    feature_count: number
    max_depth: number
    coverage: number | null
    total_tokens: number | null
  }
  domain_coverage: Record<string, DomainCoverage>
  has_deviation_map: boolean
  has_multimodal: boolean
}

export interface Artifact {
  key: string
  label: string
  file: string
  path: string
  exists: boolean
  size: number
  format: 'markdown' | 'json'
}

export interface ReportBundle {
  participant_id: string
  output_dir: string
  exists: boolean
  artifacts: Artifact[]
  performance_report: Record<string, any> | null
  patient_report: Record<string, any> | null
  execution_log: unknown
  dataflow_audit: Record<string, any> | null
  report_markdown: string
  deep_phenotype_markdown: string
  data_overview: Record<string, any> | null
  deviation_map: Record<string, any> | null
  run?: RunSummary
  cost?: CostSummary | null
}

export interface PromptRow {
  scope: 'agent' | 'tool'
  name: string
  role: string
  title: string
  description: string
  characters: number
  lines: number
  modified: boolean
  has_snapshot: boolean
}

export interface Capabilities {
  name: string
  full_name: string
  version: string
  default_model: string
  python: string
  platform: string
  stages: string[]
  agent_roles: AgentRole[]
  instruction_slots: InstructionSlot[]
  /** `has_model` is false for the Executor, which uses the tool model. */
  agents: { role: string; label: string; stage: number; summary: string; has_model: boolean }[]
  tools: { name: string; family: string; summary: string }[]
  prediction_types: { value: PredictionType; label: string; summary: string; needs: string[] }[]
  cost_model: Record<string, unknown>
  provider_links: Record<string, string>
}
