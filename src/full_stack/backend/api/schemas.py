"""
Request/response contract for the dashboard service.

Everything the web client can control lives here. The shape is deliberately
grouped by concern so the interface can present a short essential form and keep
the rest behind an "advanced" disclosure without losing any resolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from . import DEFAULT_MODEL

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
BackendName = Literal["openrouter", "openai", "local"]


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
    context_window: int = Field(0, ge=0, description="0 resolves from the catalog")
    embedding_model: str = "text-embedding-3-large"
    role_models: RoleModelMap = Field(default_factory=RoleModelMap)
    role_max_tokens: RoleTokenMap = Field(default_factory=RoleTokenMap)
    role_temperatures: RoleTemperatureMap = Field(default_factory=RoleTemperatureMap)


class EngineConfig(BaseModel):
    max_iterations: int = Field(3, ge=1, le=10)
    executor_max_workers: int = Field(12, ge=1, le=64)
    auto_repair_enabled: bool = True
    detailed_tool_logging: bool = False
    verbose: bool = True


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
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    local: LocalBackendConfig = Field(default_factory=LocalBackendConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    cost: CostGuardConfig = Field(default_factory=CostGuardConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    instructions: Dict[str, str] = Field(default_factory=dict)
    onboarding_complete: bool = False
    tour_completed: List[str] = Field(default_factory=list)


class ConfigPatch(BaseModel):
    """Partial update. Only the sections present are merged."""

    connection: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
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
    provider: Literal["openrouter", "openai"] = "openrouter"
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
    root: Optional[TaskNodeInput] = None


# --- runs --------------------------------------------------------------------


class RunOverrides(BaseModel):
    """Per-run deltas on top of the saved configuration."""

    connection: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
    token_budget: Optional[Dict[str, Any]] = None
    local: Optional[Dict[str, Any]] = None
    instructions: Optional[Dict[str, str]] = None


class RunRequest(BaseModel):
    participant_dir: str
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    generate_deep_phenotype: bool = True
    label: str = ""
    batch_id: Optional[str] = None


class BatchRequest(BaseModel):
    participant_dirs: List[str]
    task: TaskSpecInput = Field(default_factory=TaskSpecInput)
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    generate_deep_phenotype: bool = True
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
    generate_deep_phenotype: bool = True


class DeepReportRequest(BaseModel):
    focus_modalities: str = ""
    general_instruction: str = ""


class PromptUpdate(BaseModel):
    content: str
