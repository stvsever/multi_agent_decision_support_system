"""
COMPASS - Clinical Ontology-driven Multi-modal Predictive Agentic Support System

Global Configuration Settings

Centralizes all configuration including model names, token limits, 
retry settings, and file paths.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv, dotenv_values

# Runtime roots after the source tree was moved under src/full_stack.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Load environment variables from the repository root.
_ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE)


def _clean_secret(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _looks_placeholder_secret(value: str) -> bool:
    text = _clean_secret(value).lower()
    if not text:
        return True
    placeholder_markers = (
        "newkeyhere",
        "your_key",
        "your-api-key",
        "placeholder",
        "changeme",
        "replace_me",
        "replace-with",
        "set_me",
        "dummy",
        "example",
        "<",
    )
    return any(marker in text for marker in placeholder_markers)


_DOTENV_CACHE = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}


def _resolve_secret_from_env_or_dotenv(name: str) -> str:
    env_value = _clean_secret(os.getenv(name, ""))
    dotenv_value = _clean_secret(_DOTENV_CACHE.get(name, ""))
    if env_value and not _looks_placeholder_secret(env_value):
        return env_value
    if dotenv_value and not _looks_placeholder_secret(dotenv_value):
        return dotenv_value
    return env_value or dotenv_value or ""

# Suppress warnings for cleaner logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pydantic")

# Configure HuggingFace environment variables to suppress specific warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# System branding
COMPASS_FULL_NAME = "Clinical Ontology-driven Multi-modal Predictive Agentic Support System"
COMPASS_VERSION = "1.0.0"


from enum import Enum

class LLMBackend(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    LOCAL = "local"


# --- context windows ---------------------------------------------------------

#: Roles that can carry their own context window override.
CONTEXT_WINDOW_ROLES = ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool")
#: Override key that applies to every role.
ALL_ROLES_KEY = "all"
#: Used when nothing else reports a window. Conservative: an understated window
#: only means more orchestration, an overstated one a rejected prompt.
GENERIC_FALLBACK_CONTEXT_TOKENS = 128_000

#: Total context (input plus output) of common models, for when the catalog cache
#: does not list them. Keys: lower-case ids without the vendor prefix, the ":variant"
#: suffix or a quantization suffix. Values follow the providers' published limits
#: (OpenRouter list, September 2026); where a provider also caps the input
#: separately the smaller, safe value is used (gpt-5: 272K input of a 400K window).
KNOWN_CONTEXT_WINDOWS: Dict[str, int] = {
    # DeepSeek (the default model)
    "deepseek-v4-flash-0731": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    # Google
    "gemini-3.1-flash-lite": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    # OpenAI
    "gpt-5": 272_000,
    "gpt-5-mini": 272_000,
    "gpt-5-nano": 272_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    # Anthropic (standard window; the 1M window is opt-in per request)
    "claude-3.5-sonnet": 200_000,
    "claude-3.7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-sonnet-4.5": 200_000,
    "claude-opus-4": 200_000,
    "claude-opus-4.1": 200_000,
    "claude-haiku-4.5": 200_000,
    # Open weights, native window without rope scaling (local or hosted)
    "qwen3-8b": 40_960,
    "qwen3-14b": 40_960,
    "qwen3-32b": 40_960,
    "qwen2.5-0.5b-instruct": 32_768,
    "qwen2.5-7b-instruct": 32_768,
    "qwen2.5-14b-instruct": 32_768,
    "qwen2.5-32b-instruct": 32_768,
    "qwen2.5-72b-instruct": 32_768,
    "llama-3.1-8b-instruct": 131_072,
    "llama-3.3-70b-instruct": 131_072,
}

_QUANT_SUFFIX = re.compile(r"-(awq|gptq(-int[48])?|fp8|int[48]|bnb-4bit|gguf)$")


def _table_key(model_name: Optional[str]) -> str:
    """Lower-case id without "~", vendor prefix, ":variant" or quantization suffix."""
    text = str(model_name or "").strip().lower().lstrip("~")
    text = text.split(":", 1)[0]
    text = re.sub(r"^.*/", "", text)
    return _QUANT_SUFFIX.sub("", text)


def known_context_window(model_name: Optional[str]) -> int:
    """Window from the built-in table, 0 when the model is not listed."""
    return int(KNOWN_CONTEXT_WINDOWS.get(_table_key(model_name), 0) or 0)


@dataclass(frozen=True)
class ContextWindow:
    """A resolved context window and the source that decided it."""

    tokens: int
    source: str  # decision_state | role_override | local_served | catalog | known_table | configured | fallback
    model: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"tokens": int(self.tokens), "source": self.source, "model": self.model}


def parse_role_context_windows(entries) -> Dict[str, int]:
    """
    Parse "role=tokens" pairs (comma-separated, or a list of such strings).

    Roles: orchestrator, critic, predictor, integrator, communicator, tool, or
    "all". Raises ValueError with a readable message on anything else.
    """
    if entries is None:
        return {}
    items = [entries] if isinstance(entries, str) else list(entries)
    out: Dict[str, int] = {}
    allowed = set(CONTEXT_WINDOW_ROLES) | {ALL_ROLES_KEY}
    for raw in items:
        for part in str(raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"'{part}' is not ROLE=TOKENS (for example predictor=128000).")
            role, _, value = part.partition("=")
            role = role.strip().lower()
            if role not in allowed:
                raise ValueError(f"Unknown role '{role}'; use one of {', '.join(sorted(allowed))}.")
            try:
                tokens = int(str(value).strip().replace("_", ""))
            except ValueError as exc:
                raise ValueError(f"'{value.strip()}' is not a whole number of tokens for role '{role}'.") from exc
            if tokens < 1024:
                raise ValueError(f"The context window for '{role}' must be at least 1024 tokens; got {tokens}.")
            out[role] = tokens
    return out


@dataclass
class ModelConfig:
    """
    Configuration for LLM models.
    
    Public runs default to DeepSeek V4 Flash through OpenRouter.
    Role-specific models can still be overridden from the CLI or the dashboard.
    """
    # Backend Selection
    backend: LLMBackend = LLMBackend.OPENROUTER
    public_model_name: str = "deepseek/deepseek-v4-flash-0731"
    # Context window for a hosted model that neither the catalog cache nor the
    # built-in table knows (0: not configured, the generic fallback applies).
    # See Settings.context_window_resolution for the full order.
    public_max_context_tokens: int = 0
    # Forced context window per role ("orchestrator", ..., "tool", or "all" for
    # every role). It wins over every other source except a structured decision
    # model's fixed state limit.
    role_context_windows: Dict[str, int] = field(default_factory=dict)
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    local_model_name: str = "Qwen/Qwen3-14B-AWQ"  
    local_max_tokens: int = 32768  # Local-hosting default context window
    # Local backend advanced configuration
    local_backend_type: str = "vllm"
    local_dtype: str = "auto"
    local_quantization: Optional[str] = None
    local_tensor_parallel_size: int = 1
    local_pipeline_parallel_size: int = 1
    local_gpu_memory_utilization: float = 0.9
    local_max_model_len: int = 32768
    local_kv_cache_dtype: Optional[str] = "auto"  # Prefer stable runtime default; override to fp8 explicitly if desired
    local_enforce_eager: bool = False
    local_trust_remote_code: bool = True
    local_attn_implementation: str = "auto"  
    
    # Cost-efficient public profile.
    orchestrator_model: str = "deepseek/deepseek-v4-flash-0731"
    critic_model: str = "deepseek/deepseek-v4-flash-0731"
    predictor_model: str = "deepseek/deepseek-v4-flash-0731"
    integrator_model: str = "deepseek/deepseek-v4-flash-0731"
    communicator_model: str = "deepseek/deepseek-v4-flash-0731"
    tool_model: str = "deepseek/deepseek-v4-flash-0731"
    
    orchestrator_max_tokens: int = 64000
    critic_max_tokens: int = 64000
    predictor_max_tokens: int = 64000
    integrator_max_tokens: int = 64000
    communicator_max_tokens: int = 64000
    tool_max_tokens: int = 24000
    
    # Temperature settings
    orchestrator_temperature: float = 0.3
    critic_temperature: float = 0.2
    predictor_temperature: float = 0.2  # Reduced to avoid verbose loops
    integrator_temperature: float = 0.3
    communicator_temperature: float = 0.2
    tool_temperature: float = 0.5


@dataclass
class RetryConfig:
    """Configuration for retry and auto-repair logic."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    auto_repair_enabled: bool = True
    max_critic_iterations: int = 2


@dataclass
class TokenBudgetConfig:
    """
    Token budget constraints for processing.

    `explicit_*` record which values a caller set deliberately. The derivation
    helpers recompute the rest on every pipeline start, so without this an
    explicit choice would be overwritten by the derived default.
    """

    explicit_limits: Dict[str, int] = field(default_factory=dict)
    explicit_component_budgets: Dict[str, int] = field(default_factory=dict)
    explicit_role_max_tokens: Dict[str, int] = field(default_factory=dict)
    total_budget: int = 1500000
    orchestrator_budget: int = 50000
    executor_budget_per_step: int = 30000
    fusion_budget: int = 90000  # Reduced for gpt-5(-nano) safety
    integrator_budget: int = 90000 
    predictor_budget: int = 100000
    critic_budget: int = 50000
    communicator_budget: int = 100000
    
    # Granular controls
    max_agent_input_tokens: int = 30000   # Local Qwen HPC profile default (high-input)
    max_agent_output_tokens: int = 16000  # Local Qwen HPC profile default
    max_tool_input_tokens: int = 30000    # Local tool input aligned with agent input budget
    max_tool_output_tokens: int = 8000    # Local 32K profile default


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability methods."""

    enabled: bool = False
    methods: list[str] = field(default_factory=list)  # external|internal|hybrid
    run_full_validation: bool = False
    run_on_final_selected_attempt: bool = True

    # External (aHFR-TokenSHAP)
    external_k: int = 4
    external_runs: int = 1
    external_adaptive: bool = True

    # Internal (IGA)
    internal_model: str = "Qwen/Qwen3-0.5B-Instruct"
    internal_steps: int = 8
    internal_baseline_mode: str = "mask"
    internal_span_mode: str = "value"

    # Hybrid (LLM-select)
    hybrid_model: str = "deepseek/deepseek-v4-flash-0731"
    hybrid_repeats: int = 1
    hybrid_temperature: float = 0.3


ORCHESTRATION_MODES = ("auto", "always", "never")


@dataclass
class OrchestrationConfig:
    """
    When the multi-agent orchestration workflow (Orchestrator, Executor tools,
    Integrator) runs before the Predictor.

    auto:   run it only when the participant record does not fit the Predictor's
            input budget; a record that fits goes straight to the Predictor.
    always: run it on every attempt, also for records that would fit, to
            re-represent the phenotype before prediction.
    never:  never run it; a record that does not fit is packed to the budget and
            the coverage loss is recorded.

    `threshold_tokens` overrides the budget used by `auto` (0 means the
    Predictor's own input budget).
    """

    mode: str = field(default_factory=lambda: os.getenv("COMPASS_ORCHESTRATION", "auto"))
    threshold_tokens: int = 0


@dataclass
class DecisionConfig:
    """
    Settings for structured decision models (System One models such as TypeSafe
    Jev) serving the Predictor role. Every other role stays a conventional LLM.
    """

    # Transport: "openrouter" (OPENROUTER_API_KEY) or "typesafe" (TYPESAFE_API_KEY).
    provider: str = field(default_factory=lambda: os.getenv("COMPASS_DECISION_PROVIDER", "openrouter"))
    # Option orders asked per Choice. The answers are averaged, which removes the
    # first-option preference the model card documents.
    choice_orders: int = 3
    # Levels per regression Score (the API accepts at most 10).
    score_levels: int = 10
    # A second, zoomed Score pass for continuous outputs, so the estimate is not
    # limited to the width of one coarse level.
    regression_refine: bool = True
    # The decision critic rejects an attempt when a classification moves more than
    # this between two presentation orders (largest total variation distance) ...
    stability_threshold: float = 0.25
    # ... or a regression estimate moves more than this many reference standard
    # deviations between the ascending and descending level orders ...
    regression_stability_threshold: float = 0.5
    # Evidence sufficiency (a Noul asked in the same request) is always reported.
    # It only gates when this threshold is above 0: a record can be genuinely
    # uninformative for a target (a demographics-only tier, for example), and no
    # other evidence route can add information the record does not hold.
    sufficiency_threshold: float = 0.0
    # Provider tokens per cl100k token, used to keep the state inside the model's
    # limit. Measured at 1.14 on jev-1.13; 1.2 leaves a margin.
    tokenizer_ratio: float = 1.2
    request_timeout_seconds: float = 120.0
    max_retries: int = 3
    # Conventional LLM that writes the question book (label definitions and
    # output scales) once per task. Empty means the Orchestrator's model.
    compiler_model: str = ""


@dataclass
class PathConfig:
    """File and directory paths configuration."""
    base_dir: Path = field(default_factory=lambda: BACKEND_ROOT)
    agent_prompts_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "agents" / "prompts")
    tool_prompts_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "tools" / "prompts")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    overview_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "overview")
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.tool_prompts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """
    Master settings class combining all configuration sections.
    
    Usage:
        settings = get_settings()
        model = settings.models.orchestrator_model
    """
    models: ModelConfig = field(default_factory=ModelConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # API Configuration
    openai_api_key: str = field(default_factory=lambda: _resolve_secret_from_env_or_dotenv("OPENAI_API_KEY"))
    openrouter_api_key: str = field(default_factory=lambda: _resolve_secret_from_env_or_dotenv("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter_site_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", ""))
    openrouter_app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "COMPASS"))
    typesafe_api_key: str = field(default_factory=lambda: _resolve_secret_from_env_or_dotenv("TYPESAFE_API_KEY"))

    # Reasoning effort forwarded to providers that support it. Providers that do
    # not support it ignore the field. Reasoning tokens are billed as output and
    # count against the output ceiling, so on a reasoning model they can consume
    # the whole budget and truncate the answer: "off" is the safe default, and
    # "low" through "high" are available when a task needs deliberation.
    reasoning_effort: str = field(default_factory=lambda: os.getenv("COMPASS_REASONING_EFFORT", "off"))

    # Upper bound on a single provider request. Without this the SDK default
    # lets one stalled call hold a run open indefinitely.
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("COMPASS_REQUEST_TIMEOUT", "300"))
    )

    # Logging settings
    log_level: str = "INFO"
    verbose_logging: bool = True
    save_intermediate_outputs: bool = True
    detailed_tool_logging: bool = False  # New flag for capturing raw I/O

    
    # Prediction targets - Removed strict enum to allow dynamic phenotype strings
    # valid_targets: tuple = ("neuropsychiatric", "neurologic")
    
    # Data file names (expected in participant directory)
    data_overview_file: str = "data_overview.json"
    multimodal_data_file: str = "multimodal_data.json"
    non_numerical_data_file: str = "non_numerical_data.txt"
    hierarchical_deviation_file: str = "hierarchical_deviation_map.json"
    # A record may omit its deviation map: without a reference sample there are no
    # deviation scores, and the loader then derives the hierarchy from the leaves.
    optional_participant_files: tuple = ("hierarchical_deviation",)

    def _normalize_model_name(self, model_name: Optional[str]) -> str:
        return _table_key(model_name)

    def _catalog_applies(self) -> bool:
        """The OpenRouter catalog describes models served by OpenRouter (or OpenAI), not a self-hosted endpoint."""
        if self.models.backend == LLMBackend.LOCAL:
            return False
        if self.models.backend == LLMBackend.OPENAI:
            return True
        base = str(self.openrouter_base_url or "").strip().lower()
        return (not base) or "openrouter.ai" in base

    def context_window_resolution(
        self, model_name: Optional[str] = None, role: Optional[str] = None
    ) -> "ContextWindow":
        """
        The context window (total tokens, input plus output) one role works with, and where it came from.

        Order, first match wins:
          1. decision_state   a structured decision model's fixed state limit, in cl100k tokens
                              (any backend; role overrides do not apply, the provider enforces it)
          2. role_override    an explicit per-role window (CLI --role_context_window or
                              --context_window, dashboard per-role setting); "all" covers every role
          3. local_served     local backend: the served max model length (--local_max_model_len,
                              else --max_tokens), which the server enforces
          4. catalog          the cached OpenRouter model list (~/.compass/model_catalog.json),
                              only for an OpenRouter or OpenAI endpoint
          5. known_table      the built-in table of common models (KNOWN_CONTEXT_WINDOWS)
          6. configured       the explicitly configured window for other hosted models
                              (--public_max_context_tokens, dashboard "Context window")
          7. fallback         GENERIC_FALLBACK_CONTEXT_TOKENS

        If the provider's real window is smaller than the resolved one, a direct
        prompt is rejected as too long and the pipeline orchestrates that attempt.
        """
        requested = str(model_name or "").strip() or str(self.models.public_model_name or "").strip()

        # 1. A structured decision model has a small, fixed state limit counted in
        # its own tokens. Report it in the cl100k tokens the engine counts with.
        from ..agents.decision.registry import get_decision_model_spec

        decision_spec = get_decision_model_spec(requested)
        if decision_spec is not None:
            return ContextWindow(
                decision_spec.state_budget_cl100k(ratio=float(self.decision.tokenizer_ratio)),
                "decision_state",
                requested,
            )

        # 2. Explicit per-role override.
        overrides = {str(k).strip().lower(): int(v or 0) for k, v in (self.models.role_context_windows or {}).items()}
        role_key = str(role or "").strip().lower()
        forced = (overrides.get(role_key, 0) if role_key else 0) or overrides.get(ALL_ROLES_KEY, 0)
        if forced > 0:
            return ContextWindow(max(1024, forced), "role_override", requested)

        # 3. Local backend: the served length is a hard limit.
        if self.models.backend == LLMBackend.LOCAL:
            local_len = int(getattr(self.models, "local_max_model_len", 0) or 0)
            if local_len > 0:
                return ContextWindow(local_len, "local_served", requested)
            return ContextWindow(
                max(1024, int(getattr(self.models, "local_max_tokens", 32768) or 32768)), "local_served", requested
            )

        # 4. The cached provider catalog is the most current source.
        if self._catalog_applies():
            catalog_ctx = _catalog_context_length(requested)
            if catalog_ctx:
                return ContextWindow(catalog_ctx, "catalog", requested)

        # 5. Built-in table.
        known = known_context_window(requested)
        if known:
            return ContextWindow(known, "known_table", requested)

        # 6. Explicitly configured window for other hosted models.
        configured = int(getattr(self.models, "public_max_context_tokens", 0) or 0)
        if configured > 0:
            return ContextWindow(max(8192, configured), "configured", requested)

        # 7. Generic fallback.
        return ContextWindow(GENERIC_FALLBACK_CONTEXT_TOKENS, "fallback", requested)

    def effective_context_window(self, model_name: Optional[str] = None, role: Optional[str] = None) -> int:
        """Context window in tokens; see context_window_resolution for the order of sources."""
        return int(self.context_window_resolution(model_name=model_name, role=role).tokens)

    def role_context_windows(self) -> Dict[str, "ContextWindow"]:
        """The resolved window of every role's current model."""
        return {
            role: self.context_window_resolution(getattr(self.models, f"{role}_model", None), role=role)
            for role in CONTEXT_WINDOW_ROLES
        }

    def auto_output_token_limit(self, model_name: Optional[str] = None, role: Optional[str] = None) -> int:
        ctx = self.effective_context_window(model_name=model_name, role=role)
        return max(1024, min(64000, int(ctx * 0.5)))
    
    def validate(self) -> bool:
        """Validate that required settings are present."""
        if self.models.backend == LLMBackend.LOCAL:
            return True
        if self.models.backend == LLMBackend.OPENROUTER:
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            return True
        if self.models.backend == LLMBackend.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return True
        return True
    
    def get_participant_files(self, participant_dir: Path) -> Dict[str, Path]:
        """Get paths to all expected participant files."""
        return {
            "data_overview": participant_dir / self.data_overview_file,
            "multimodal_data": participant_dir / self.multimodal_data_file,
            "non_numerical_data": participant_dir / self.non_numerical_data_file,
            "hierarchical_deviation": participant_dir / self.hierarchical_deviation_file,
        }

    def get_required_participant_files(self, participant_dir: Path) -> Dict[str, Path]:
        """The participant files a record cannot do without."""
        return {
            key: path
            for key, path in self.get_participant_files(participant_dir).items()
            if key not in self.optional_participant_files
        }


def catalog_cache_path() -> Path:
    """The OpenRouter model list the dashboard caches (and `main.py --refresh_catalog` writes)."""
    override = os.getenv("COMPASS_HOME", "").strip()
    return (Path(override).expanduser() if override else Path.home() / ".compass") / "model_catalog.json"


def catalog_cache_status() -> Dict[str, object]:
    """Whether the catalog cache exists and how old it is, for the configuration check."""
    import json
    import time

    path = catalog_cache_path()
    status: Dict[str, object] = {"path": str(path), "present": path.exists(), "age_hours": None, "models": 0}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            fetched = float(payload.get("fetched_at") or path.stat().st_mtime)
            status["age_hours"] = round(max(0.0, time.time() - fetched) / 3600.0, 1)
            status["models"] = len(payload.get("models") or [])
        except Exception:
            status["present"] = False
    return status


def _catalog_context_length(model_name: Optional[str]) -> int:
    """Context window from the dashboard's cached OpenRouter catalog, 0 when unknown."""
    if not model_name:
        return 0
    path = catalog_cache_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0
    cached = _CATALOG_CONTEXT_CACHE.get("index")
    stamp = (str(path), mtime)
    if cached is None or _CATALOG_CONTEXT_CACHE.get("stamp") != stamp:
        try:
            import json

            rows = json.loads(path.read_text()).get("models") or []
            cached = {
                str(r.get("id") or "").lower(): int(r.get("context_length") or 0)
                for r in rows
                if isinstance(r, dict)
            }
        except Exception:
            cached = {}
        _CATALOG_CONTEXT_CACHE.update({"index": cached, "stamp": stamp})
    for key in _catalog_candidates(model_name):
        value = int(cached.get(key, 0) or 0)
        if value:
            return value
    return 0


def _catalog_candidates(model_name: str) -> list:
    """Catalog ids to try, most specific first: as given, without "~", without ":variant", vendor-prefixed."""
    text = str(model_name or "").strip().lower()
    candidates = [text, text.lstrip("~")]
    for item in list(candidates):
        if ":" in item:
            candidates.append(item.split(":", 1)[0])
    base = candidates[-1]
    if "/" not in base and base.startswith(("gpt-", "o1", "o3", "o4")):
        # The OpenRouter client sends bare OpenAI ids as openai/<id>.
        candidates.append(f"openai/{base}")
    seen = []
    for item in candidates:
        if item and item not in seen:
            seen.append(item)
    return seen


_CATALOG_CONTEXT_CACHE: Dict[str, object] = {}


def predictor_output_reserve(max_completion_tokens: int) -> int:
    """Tokens the LLM Predictor keeps free for its answer (between 1024 and 8192)."""
    return max(1024, min(8192, int(max_completion_tokens) + 1024))


def llm_predictor_input_budget(settings: "Settings", *, model: Optional[str] = None, max_completion_tokens: Optional[int] = None) -> int:
    """
    The input an LLM Predictor accepts, in cl100k tokens: its context window
    minus the output reserve, capped by `max_agent_input_tokens` when set.

    The engine routes on this number and the dashboard estimate calls the same
    function, so the two cannot drift apart.
    """
    model = model or settings.models.predictor_model
    completion = int(max_completion_tokens if max_completion_tokens is not None else (settings.models.predictor_max_tokens or 4096))
    configured = int(getattr(settings.token_budget, "max_agent_input_tokens", 0) or 0)
    context_window = int(settings.effective_context_window(model, role="predictor"))
    by_context = max(2048, context_window - predictor_output_reserve(completion or 4096))
    if configured > 0:
        return max(2048, min(configured, by_context))
    return by_context


# Singleton pattern for settings
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Force reload of settings (useful for testing)."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance


# Print configuration on import if verbose
if __name__ == "__main__":
    settings = get_settings()
    print("=" * 60)
    print("COMPASS Configuration")
    print("=" * 60)
    print(f"Orchestrator Model: {settings.models.orchestrator_model}")
    print(f"Tool Model: {settings.models.tool_model}")
    print(f"Total Token Budget: {settings.token_budget.total_budget}")
    print(f"Max Critic Iterations: {settings.retry.max_critic_iterations}")
    api_ok = settings.openrouter_api_key if settings.models.backend == LLMBackend.OPENROUTER else settings.openai_api_key
    print(f"API Key Present: {'Yes' if api_ok else 'No'}")
    print("=" * 60)
