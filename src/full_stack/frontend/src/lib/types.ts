/** Types mirroring the dashboard service contract. */

export type Theme = 'system' | 'light' | 'dark'
export type Accent = 'indigo' | 'violet' | 'teal' | 'amber' | 'rose' | 'slate'
export type Density = 'comfortable' | 'compact'
export type BackendName = 'openrouter' | 'local'
export type CredentialProvider = 'openrouter' | 'huggingface'
export type LocalRuntime = 'native' | 'docker' | 'apptainer'
export type LocalScheduler = 'none' | 'slurm'
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

/**
 * Self-hosted inference. The fields past `trust_remote_code` describe where the
 * weights actually run, which is what makes a multi-GPU or cluster deployment
 * expressible rather than only a single local process.
 */
export interface LocalBackendConfig {
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
  runtime: LocalRuntime
  image: string
  gpu_count: number
  scheduler: LocalScheduler
  slurm_partition: string
  slurm_account: string
  slurm_time: string
  slurm_nodes: number
  slurm_gpus_per_node: number
  slurm_cpus_per_task: number
  slurm_mem_gb: number
}

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
  local: LocalBackendConfig
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
  /** One-off explanations for a configuration the service had to heal on load. */
  notices?: string[]
}

/**
 * Reachability. The interface stays silent when everything is fine and only
 * marks the failure, so a healthy state costs no screen space.
 */
export interface Connectivity {
  online: boolean
  provider_reachable: boolean
  checked_at: number
  reason: string
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

/** A directory that holds some but not all of the four required input files. */
export interface NearMiss {
  directory: string
  present: string[]
  missing: string[]
}

export interface ScannedRoot {
  root: string
  label: string
  found: number
  bundled: boolean
  scanned_dir_count?: number
  near_misses?: NearMiss[]
}

export interface ParticipantsResponse {
  participants: Participant[]
  roots: ScannedRoot[]
  count: number
  added?: boolean
  already_present?: boolean
}

export interface BrowseEntry {
  name: string
  path: string
  is_dir: boolean
  is_participant: boolean
  child_dir_count: number
}

export interface BrowseResponse {
  path: string
  parent: string | null
  entries: BrowseEntry[]
  roots: { label: string; path: string }[]
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
  audit_summary?: AuditSummary | null
}

/** What a structural audit produces. No model is called, so there is no cost. */
export interface AuditSummary {
  predictor_payload_tokens: number
  chunk_budget_tokens: number
  chunk_count: number
  predictor_input_mode: string
  assertions_ok: boolean
  coverage_summary: Record<string, unknown>
  section_stats?: { name: string; tokens: number; feature_key_count: number }[]
  chunk_stats?: { chunk_index: number; sections: string[]; tokens: number }[]
  assertions?: Record<string, boolean>
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
  aggregate?: AggregateStats
}

/** Present on a merged cohort tree; absent on a single participant's tree. */
export interface AggregateStats {
  present_in: number
  participant_count: number
  coverage: number
  membership: 'shared' | 'common' | 'partial'
  n: number
  mean: number | null
  sd: number | null
  min: number | null
  max: number | null
  q1: number | null
  median: number | null
  q3: number | null
}

export interface Ontology {
  participant_id: string
  participant_ids?: string[]
  aggregated?: boolean
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


/* --- Cohort distributions -------------------------------------------------- */

export type VariableType = 'continuous' | 'integer' | 'ordinal' | 'nominal' | 'missing'

export interface HistogramBin {
  start: number
  end: number
  count: number
}

export interface CategoryCount {
  label: string
  count: number
  proportion: number
}

export interface NumericSummary {
  n: number
  mean: number | null
  sd: number | null
  min: number | null
  max: number | null
  median: number | null
  q1: number | null
  q3: number | null
  bins: HistogramBin[]
}

export interface DistributionResponse {
  path: string[]
  label: string
  variable_type: VariableType
  participant_count: number
  values: (number | string | boolean | null)[]
  numeric?: NumericSummary
  categories?: CategoryCount[]
  z_scores?: NumericSummary
  rows: { participant_id: string; value: number | string | boolean | null; z_score: number | null }[]
}

/* --- Self-hosted deployment ------------------------------------------------ */

export interface DeployCommand {
  runtime: LocalRuntime | 'slurm'
  title: string
  command: string
  note?: string
}

export interface DeployPlan {
  engine: 'vllm' | 'transformers'
  model_name: string
  gpu_count: number
  tensor_parallel_size: number
  pipeline_parallel_size: number
  estimated_vram_gb: number | null
  estimate_basis: string
  warnings: string[]
  commands: DeployCommand[]
  endpoint_hint: string
}

export interface DeployProbe {
  cuda_available: boolean
  gpu_count: number
  gpu_names: string[]
  total_vram_gb: number | null
  vllm_installed: boolean
  transformers_installed: boolean
  docker_available: boolean
  apptainer_available: boolean
  slurm_available: boolean
}

export interface HfModelRow {
  id: string
  downloads: number
  likes: number
  pipeline_tag: string
  tags: string[]
  gated: boolean
  is_embedding: boolean
}

export interface HfModelDetail extends Partial<HfModelRow> {
  id: string
  context_length: number | null
  architectural_context_length?: number | null
  parameter_count?: number | null
  license?: string
  model_type?: string
  architecture?: string[]
  error?: string
}
