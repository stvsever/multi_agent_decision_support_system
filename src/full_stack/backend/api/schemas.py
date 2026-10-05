"""
Request/response contract for the dashboard service.

Everything the web client can control lives here. The shape is deliberately
grouped by concern so the interface can present a short essential form and keep
the rest behind an "advanced" disclosure without losing any resolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..agents.decision import decision_role_conflicts, is_decision_model
from . import DEFAULT_MODEL


def readable_validation_error(exc: ValidationError) -> str:
    """
    A rejected request in the words the constraint was written in.

    The model validators here raise sentences meant for the person editing the
    form, and pydantic prefixes each one with "Value error", which turns it into
    something that reads like a defect report. Every screen that surfaces a
    rejection goes through this, so all of them say the same thing.
    """
    parts: List[str] = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error.get("loc", ()) if p not in ("__root__", "body"))
        message = str(error.get("msg", "invalid value")).replace("Value error, ", "")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "Invalid configuration."

AGENT_ROLES = ("orchestrator", "integrator", "predictor", "critic", "communicator", "tool")
INSTRUCTION_SLOTS = (
    "global",
    "orchestrator",
    "executor",
    "tools",
    "integrator",
    "predictor",
    "critic",
    "communicator",
)

PredictionType = Literal[
    "binary",
    "multiclass",
    "regression_univariate",
    "regression_multivariate",
    "hierarchical",
]
BackendName = Literal["openrouter", "local"]
OrchestrationMode = Literal["auto", "always", "never"]
DecisionProvider = Literal["openrouter", "typesafe"]

#: Providers the dashboard can store a key for. HuggingFace is here because
#: gated weight repositories need a token before the local runtime can pull them,
#: and TypeSafe because its native route serves structured decision models.
CredentialProvider = Literal["openrouter", "huggingface", "typesafe"]


# --- configuration -----------------------------------------------------------


class RoleModelMap(BaseModel):
    """Per-role model override. Empty string means "use the default model"."""

    orchestrator: str = ""
    integrator: str = ""
    predictor: str = ""
    critic: str = ""
    communicator: str = ""
    tool: str = ""


class RoleTokenMap(BaseModel):
    """
    Per-role output-token ceiling. Zero means "derive from the context window".

    The defaults are sized to the job each role actually does. Deriving them
    from a million-token context window instead produces a 64k ceiling for every
    role, which makes plans slow and expensive for no benefit.
    """

    orchestrator: int = 8_000
    integrator: int = 8_000
    predictor: int = 12_000
    critic: int = 6_000
    communicator: int = 16_000
    tool: int = 4_000


class RoleWindowMap(BaseModel):
    """
    Per-role context window override in tokens. Zero means "resolve it": the
    catalog, then the built-in table, then the configured window. A structured
    decision model keeps its fixed state limit whatever is set here.
    """

    orchestrator: int = Field(0, ge=0, le=10_000_000)
    integrator: int = Field(0, ge=0, le=10_000_000)
    predictor: int = Field(0, ge=0, le=10_000_000)
    critic: int = Field(0, ge=0, le=10_000_000)
    communicator: int = Field(0, ge=0, le=10_000_000)
    tool: int = Field(0, ge=0, le=10_000_000)

    @model_validator(mode="after")
    def _at_least_a_small_window(self) -> "RoleWindowMap":
        small = [role for role, value in self.model_dump().items() if 0 < int(value) < 1024]
        if small:
            raise ValueError(f"A context window override must be 0 (resolve) or at least 1024 tokens: {', '.join(small)}.")
        return self


class RoleTemperatureMap(BaseModel):
    orchestrator: float = 0.3
    integrator: float = 0.3
    predictor: float = 0.2
    critic: float = 0.2
    communicator: float = 0.2
    tool: float = 0.5


class ConnectionConfig(BaseModel):
    backend: BackendName = "openrouter"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = ""
    openrouter_app_name: str = "COMPASS"
    request_timeout_seconds: int = Field(180, ge=10, le=1800)
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay_seconds: float = Field(1.0, ge=0.0, le=60.0)


class ModelConfig(BaseModel):
    default_model: str = DEFAULT_MODEL
    # Measured on the default model: reasoning off runs about four times faster
    # and three times cheaper, and stops reasoning tokens from consuming the
    # entire output budget. Raise it when a task needs deeper deliberation.
    reasoning_effort: Literal["provider_default", "off", "low", "medium", "high"] = "off"
    # Window for a hosted model that neither the catalog nor the built-in table
    # knows; 0 leaves the generic fallback. Per-role overrides force a window.
    context_window: int = Field(0, ge=0, description="0: resolve from the catalog and the built-in table")
    embedding_model: str = "text-embedding-3-large"
    role_models: RoleModelMap = Field(default_factory=RoleModelMap)
    role_max_tokens: RoleTokenMap = Field(default_factory=RoleTokenMap)
    role_context_windows: RoleWindowMap = Field(default_factory=RoleWindowMap)
    role_temperatures: RoleTemperatureMap = Field(default_factory=RoleTemperatureMap)


class EngineConfig(BaseModel):
    max_iterations: int = Field(3, ge=1, le=10)
    executor_max_workers: int = Field(12, ge=1, le=64)
    auto_repair_enabled: bool = True
    detailed_tool_logging: bool = False
    verbose: bool = True
    # auto: orchestrate only when the record does not fit the Predictor input;
    # always: orchestrate every attempt; never: always hand the record over directly.
    orchestration_mode: OrchestrationMode = "auto"
    orchestration_threshold_tokens: int = Field(0, ge=0, description="0 uses the Predictor input budget")


class DecisionConfig(BaseModel):
    """
    Structured decision models (TypeSafe Jev) in the Predictor role.

    Such a model writes no text: it answers typed questions with calibrated
    probabilities, so these settings shape the questions rather than a prompt.
    """

    provider: DecisionProvider = "openrouter"
    # Option orders asked per Choice; averaging them removes the first-option preference.
    choice_orders: int = Field(3, ge=1, le=6)
    score_levels: int = Field(10, ge=2, le=10)
    regression_refine: bool = True
    # Largest probability shift between two presentation orders (classification).
    stability_threshold: float = Field(0.25, ge=0.05, le=0.6)
    # Largest gap between ascending and descending level orders, in reference SDs (regression).
    regression_stability_threshold: float = Field(0.5, ge=0.1, le=2.0)
    sufficiency_threshold: float = Field(0.0, ge=0.0, le=0.9)
    compiler_model: str = Field("", description="Blank uses the Orchestrator model")


class TokenBudgetConfig(BaseModel):
    total_budget: int = Field(1_500_000, ge=0)
    max_agent_input_tokens: int = Field(0, ge=0, description="0 derives from the context window")
    max_agent_output_tokens: int = Field(0, ge=0)
    max_tool_input_tokens: int = Field(0, ge=0)
    max_tool_output_tokens: int = Field(0, ge=0)
    orchestrator_budget: int = Field(0, ge=0)
    executor_budget_per_step: int = Field(0, ge=0)
    fusion_budget: int = Field(0, ge=0)
    integrator_budget: int = Field(0, ge=0)
    predictor_budget: int = Field(0, ge=0)
    critic_budget: int = Field(0, ge=0)
    communicator_budget: int = Field(0, ge=0)


class LocalBackendConfig(BaseModel):
    """
    Open weights served by this project, from one workstation to an HPC cluster.

    The model settings describe what to load; the runtime and scheduler settings
    describe where it runs, which is what the deployment planner turns into
    copy-pasteable commands.
    """

    model_name: str = "Qwen/Qwen3-14B-AWQ"
    max_tokens: int = Field(32768, ge=1024)
    engine: Literal["auto", "vllm", "transformers"] = "auto"
    dtype: str = "auto"
    quantization: str = ""
    kv_cache_dtype: str = "auto"
    attn_implementation: str = "auto"
    tensor_parallel_size: int = Field(1, ge=1, le=16)
    pipeline_parallel_size: int = Field(1, ge=1, le=16)
    gpu_memory_utilization: float = Field(0.9, gt=0.0, le=1.0)
    max_model_len: int = Field(0, ge=0)
    enforce_eager: bool = False
    trust_remote_code: bool = True

    runtime: Literal["native", "docker", "apptainer"] = "native"
    image: str = Field("", description="Container image or .sif path; blank uses the shipped default")
    gpu_count: int = Field(1, ge=1, le=64)
    scheduler: Literal["none", "slurm"] = "none"
    slurm_partition: str = ""
    slurm_account: str = ""
    slurm_time: str = "04:00:00"
    slurm_nodes: int = Field(1, ge=1, le=64)
    slurm_gpus_per_node: int = Field(1, ge=1, le=16)
    slurm_cpus_per_task: int = Field(8, ge=1, le=256)
    slurm_mem_gb: int = Field(0, ge=0, description="0 lets the scheduler decide")

    @model_validator(mode="after")
    def _parallelism_fits_the_gpus(self) -> "LocalBackendConfig":
        # vLLM claims one GPU per tensor-parallel rank per pipeline stage, so a
        # product larger than the machine has fails at load time with an error
        # that says nothing about which setting caused it.
        needed = self.tensor_parallel_size * self.pipeline_parallel_size
        if needed > self.gpu_count:
            raise ValueError(
                f"Tensor parallel {self.tensor_parallel_size} times pipeline parallel "
                f"{self.pipeline_parallel_size} needs {needed} GPUs, but only {self.gpu_count} "
                "are declared. Raise the GPU count or lower the parallelism."
            )
        return self


class BatchConfig(BaseModel):
    concurrency: int = Field(2, ge=1, le=16)
    continue_on_error: bool = True
    run_timeout_seconds: int = Field(3600, ge=60, le=86400)


class CostGuardConfig(BaseModel):
    warn_above_usd: float = Field(1.0, ge=0.0)
    block_above_usd: float = Field(25.0, ge=0.0, description="0 disables the hard stop")
    currency_decimals: int = Field(4, ge=2, le=6)


class WorkspaceConfig(BaseModel):
    data_roots: List[str] = Field(default_factory=list)
    output_dir: str = ""
    auto_open_report: bool = True


class AppearanceConfig(BaseModel):
    theme: Literal["system", "light", "dark"] = "system"
    accent: Literal["indigo", "violet", "teal", "amber", "rose", "slate"] = "indigo"
    density: Literal["comfortable", "compact"] = "comfortable"
    font_scale: float = Field(1.0, ge=0.85, le=1.3)
    reduced_motion: bool = False
    show_advanced_by_default: bool = False
    flow_animate_edges: bool = True
    flow_direction: Literal["LR", "TB"] = "LR"
    flow_edge_style: Literal["bezier", "smoothstep", "straight"] = "bezier"
    flow_show_tokens: bool = True
    numeric_locale: str = "en-US"


class DashboardConfig(BaseModel):
    """The complete, persisted control surface."""

    version: int = 1
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    local: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    cost: CostGuardConfig = Field(default_factory=CostGuardConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    instructions: Dict[str, str] = Field(default_factory=dict)
    onboarding_complete: bool = False
    tour_completed: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _local_backend_needs_a_model(self) -> "DashboardConfig":
        # Without this the interface happily stores a backend it cannot run and
        # the first sign of trouble is a worker traceback minutes later.
        if self.connection.backend == "local" and not self.local.model_name.strip():
            raise ValueError("Choose a local model before switching the backend to Local.")
        return self

    @model_validator(mode="after")
    def _decision_models_only_predict(self) -> "DashboardConfig":
        conflicts = decision_model_conflicts(self)
        if conflicts:
            raise ValueError(
                "Structured decision models can only serve the Predictor role. "
                f"Choose a conventional LLM for: {', '.join(conflicts)}."
            )
        return self


def decision_model_conflicts(config: "DashboardConfig") -> List[str]:
    """
    Every setting that names a structured decision model where a conventional
    LLM is needed: the default model (it fills every unset role), any role
    other than the Predictor, and the model that writes the question book.
    """
    conflicts: List[str] = []
    if is_decision_model(config.models.default_model):
        conflicts.append("default model")
    role_models = {role: getattr(config.models.role_models, role, "") for role in AGENT_ROLES}
    conflicts.extend(decision_role_conflicts(role_models))
    if is_decision_model(config.decision.compiler_model):
        conflicts.append("question book compiler")
    return conflicts


class ConfigPatch(BaseModel):
    """Partial update. Only the sections present are merged."""

    connection: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    token_budget: Optional[Dict[str, Any]] = None
    local: Optional[Dict[str, Any]] = None
    batch: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None
    workspace: Optional[Dict[str, Any]] = None
    appearance: Optional[Dict[str, Any]] = None
    instructions: Optional[Dict[str, str]] = None
    onboarding_complete: Optional[bool] = None
    tour_completed: Optional[List[str]] = None


class CredentialUpdate(BaseModel):
    provider: CredentialProvider = "openrouter"
    api_key: str = ""


class CredentialStatus(BaseModel):
    provider: str
    configured: bool
    masked: str = ""
    source: Literal["stored", "environment", "none"] = "none"


# --- task specification ------------------------------------------------------


class TaskNodeInput(BaseModel):
    node_id: str
    display_name: str
    mode: Literal[
        "binary_classification",
        "multiclass_classification",
        "univariate_regression",
        "multivariate_regression",
    ]
    class_labels: List[str] = Field(default_factory=list)
    regression_outputs: List[str] = Field(default_factory=list)
    unit_by_output: Dict[str, str] = Field(default_factory=dict)
    # Optional scale per regression output and definition per class label; a
    # structured decision model reads them, an LLM Predictor sees them in the task.
    output_scales: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    class_definitions: Dict[str, str] = Field(default_factory=dict)
    required: bool = True
    children: List["TaskNodeInput"] = Field(default_factory=list)


TaskNodeInput.model_rebuild()


class TaskSpecInput(BaseModel):
    """
    How the run defines its prediction target.

    Flat modes fill the label fields; `hierarchical` supplies a `root` tree.
    """

    prediction_type: PredictionType = "binary"
    target_label: str = "target_phenotype"
    control_label: str = "non_target_comparator"
    class_labels: List[str] = Field(default_factory=list)
    regression_outputs: List[str] = Field(default_factory=list)
    # Optional range per regression output for flat regression tasks
    # ({output: {"min": .., "max": .., "integer": ..}}).
    output_scales: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    root: Optional[TaskNodeInput] = None


# --- runs --------------------------------------------------------------------


class RunOverrides(BaseModel):
    """Per-run deltas on top of the saved configuration."""

    connection: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    token_budget: Optional[Dict[str, Any]] = None
    local: Optional[Dict[str, Any]] = None
    instructions: Optional[Dict[str, str]] = None


class RunRequest(BaseModel):
    participant_dir: str
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    generate_deep_phenotype: bool = False
    label: str = ""
    batch_id: Optional[str] = None


class BatchRequest(BaseModel):
    participant_dirs: List[str]
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    generate_deep_phenotype: bool = False
    concurrency: Optional[int] = None
    continue_on_error: Optional[bool] = None
    label: str = ""


class AuditRequest(BaseModel):
    participant_dir: str
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)


class CostEstimateRequest(BaseModel):
    participant_dirs: List[str] = Field(default_factory=list)
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    generate_deep_phenotype: bool = False


class OntologyAggregateRequest(BaseModel):
    directories: List[str] = Field(default_factory=list)


class OntologyDistributionRequest(BaseModel):
    directories: List[str] = Field(default_factory=list)
    path: List[str] = Field(default_factory=list)


class DeepReportRequest(BaseModel):
    focus_modalities: str = ""
    general_instruction: str = ""


class PromptUpdate(BaseModel):
    content: str
