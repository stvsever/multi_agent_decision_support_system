"""
Cost projection for a COMPASS run.

The per-agent token model below is a linear fit against the token ledgers of
completed runs (``performance_report_*.json``), so the projection tracks the
size of the participant's own evidence rather than a flat guess. Every
coefficient is exposed through :func:`model_profile` so the estimate can be
inspected instead of trusted blindly.

The projection follows the evidence route the engine will take. A record that
fits the Predictor's input goes to it directly, with no Orchestrator and no tool
steps; a larger one is orchestrated first. A structured decision model in the
Predictor role bills its input only and answers a question book that a
conventional LLM compiles once per task.

Route and Predictor input come from the engine itself: the configuration is
stamped onto an isolated engine Settings object (`engine_bridge.build_engine_settings`)
and the record is measured offline by `utils.core.route_preview.measure_record`,
the same DataLoader, direct executor output and `measure_direct` a run performs
before its first attempt. The context windows, the LLM input budget
(`config.settings.llm_predictor_input_budget`) and the decision state budget are
therefore the engine's numbers, not a copy of its arithmetic. Only when a record
cannot be read does the projection fall back to the overview size.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.decision import get_decision_model_spec, is_decision_model
from ..config.settings import GENERIC_FALLBACK_CONTEXT_TOKENS
from .catalog import pricing_index
from .config_store import resolve_predictor_kind
from .schemas import DashboardConfig

#: Plan size the orchestrator tends to choose. It grows with the number of
#: available domains, so it is derived per participant rather than fixed.
DEFAULT_PLAN_STEPS = 12
STEPS_BASE = 4.0
STEPS_PER_DOMAIN = 1.8


@dataclass(frozen=True)
class AgentProfile:
    """
    Fitted against the token ledgers of completed runs.

    `prompt_base` is the size-independent cost, `prompt_per_input` scales with
    the participant's `data_overview.total_tokens`, and `scales_with_plan` marks
    the roles whose payload grows with the number of tool outputs feeding them.
    """

    role: str
    calls_per_iteration: float
    prompt_base: int
    prompt_per_input: float
    completion_base: int
    completion_per_input: float = 0.0
    scales_with_plan: bool = False


AGENT_PROFILES: Dict[str, AgentProfile] = {
    "orchestrator": AgentProfile("orchestrator", 1, 4_600, 0.0, 2_000),
    "tool": AgentProfile("tool", DEFAULT_PLAN_STEPS, 1_500, 0.18, 950),
    "integrator": AgentProfile("integrator", 0, 6_000, 0.9, 900),
    # The predictor reads every tool output, so its prompt grows with plan size.
    "predictor": AgentProfile("predictor", 1, 11_500, 2.4, 2_100, scales_with_plan=True),
    "critic": AgentProfile("critic", 1, 200, 0.0, 50),
}

#: The deep phenotype report is a single extra pass after the loop, and it is
#: the largest single call in a run because it reads the whole evidence chain.
COMMUNICATOR_PROFILE = AgentProfile("communicator", 1, 25_000, 6.0, 6_000, scales_with_plan=True)


def expected_plan_steps(domain_count: int) -> int:
    """How many tool steps a plan over this many domains usually contains."""
    if domain_count <= 0:
        return DEFAULT_PLAN_STEPS
    return max(4, min(28, round(STEPS_BASE + STEPS_PER_DOMAIN * domain_count)))


#: Reported spread, since plan size and critic retries are genuinely variable.
ESTIMATE_SPREAD = 0.35

#: A structured decision model reads one packed state per request and makes at
#: most two requests per attempt (a coarse round, then a zoomed regression round).
DECISION_REQUESTS_PER_ATTEMPT = 2
#: The typed questions of one request, on top of the state.
DECISION_QUESTION_TOKENS = 10_000
#: What the state budget keeps back for the longest single question plus the
#: engine's margin (a 10-level Score is about 300 to 550 tokens, the margin 256).
DECISION_STATE_QUESTION_RESERVE = 800
#: Provider tokens per cl100k token, as the engine's decision settings assume.
DECISION_TOKENIZER_RATIO = 1.2
#: The question book is compiled once per task by a conventional LLM.
QUESTION_BOOK_PROFILE = AgentProfile("question_book", 1, 4_000, 0.0, 1_600)

#: The engine's fallback when nothing reports the Predictor's context window
#: (kept as a name for callers; the engine's own constant decides).
FALLBACK_CONTEXT_TOKENS = GENERIC_FALLBACK_CONTEXT_TOKENS
ROUTES = ("direct", "orchestrated")

#: Measurements of recent (record, configuration, task) combinations; the Studio
#: re-estimates on every settled edit, and a record does not change in between.
_MEASURE_CACHE: "OrderedDict[str, Optional[Dict[str, Any]]]" = OrderedDict()
_MEASURE_CACHE_SIZE = 512
_MEASURE_LOCK = threading.RLock()


def participant_input_tokens(participant_dir: Path) -> int:
    """Evidence volume for a participant, straight from its own overview file."""
    overview = participant_dir / "data_overview.json"
    if not overview.exists():
        return 0
    try:
        payload = json.loads(overview.read_text() or "{}")
    except json.JSONDecodeError:
        return 0
    total = payload.get("total_tokens")
    if isinstance(total, (int, float)) and total > 0:
        return int(total)
    coverage = payload.get("domain_coverage") or {}
    return int(sum(int((v or {}).get("total_tokens") or 0) for v in coverage.values()))


def participant_domain_count(participant_dir: Path) -> int:
    """Domain count drives how many tool steps the orchestrator schedules."""
    overview = participant_dir / "data_overview.json"
    if not overview.exists():
        return 0
    try:
        payload = json.loads(overview.read_text() or "{}")
    except json.JSONDecodeError:
        return 0
    available = payload.get("available_domains")
    if isinstance(available, list) and available:
        return len(available)
    coverage = payload.get("domain_coverage") or {}
    return len(coverage)


def _price(models: Dict[str, Dict[str, Any]], model_id: str) -> Dict[str, Optional[float]]:
    row = models.get(model_id) or {}
    prompt = row.get("prompt_usd_per_mtok")
    completion = row.get("completion_usd_per_mtok")
    context = row.get("context_length")
    spec = get_decision_model_spec(model_id)
    if spec is not None:
        # A decision model is priced by the registry when the catalog does not
        # carry it, and its output is free: a missing completion price is zero,
        # never a reason to call the spend unpriced.
        prompt = spec.input_price_per_million if prompt is None else prompt
        completion = spec.output_price_per_million if completion is None else completion
        context = spec.state_context_tokens
    return {
        "prompt": prompt,
        "completion": completion,
        "context_length": context,
        "known": bool(row) or spec is not None,
    }


def _role_model(config: DashboardConfig, role: str) -> str:
    explicit = str(getattr(config.models.role_models, role, "") or "").strip()
    return explicit or config.models.default_model


def engine_settings(config: DashboardConfig) -> Any:
    """The engine Settings a run with this configuration uses, built in isolation."""
    from .engine_bridge import build_engine_settings

    return build_engine_settings(config)


def predictor_input_budget(
    config: DashboardConfig,
    predictor_kind: str,
    prices: Optional[Dict[str, Dict[str, Any]]] = None,
    settings: Any = None,
) -> int:
    """
    The input the Predictor accepts, in the cl100k tokens the engine counts.

    The engine's own arithmetic on the engine's own settings: for an LLM,
    `llm_predictor_input_budget` (the role's context window minus the output
    reserve, capped by the agent input limit); for a decision model, its state
    limit in cl100k tokens minus what the longest question needs (the exact
    figure needs the task's questions, so a measured record replaces it).
    `prices` is accepted for compatibility; windows come from the engine.
    """
    from ..config.settings import llm_predictor_input_budget

    settings = settings if settings is not None else engine_settings(config)
    model_id = settings.models.predictor_model
    if predictor_kind == "decision":
        spec = get_decision_model_spec(model_id) or get_decision_model_spec(_role_model(config, "predictor"))
        if spec is None:
            return 0
        state = spec.state_budget_cl100k(ratio=float(settings.decision.tokenizer_ratio or DECISION_TOKENIZER_RATIO))
        return max(0, state - DECISION_STATE_QUESTION_RESERVE)
    if is_decision_model(model_id):
        # A what-if LLM estimate for a decision Predictor: the companion model.
        model_id = settings.models.orchestrator_model
    return llm_predictor_input_budget(settings, model=model_id)


def _measure_key(config: DashboardConfig, directory: Path, task_json: str) -> str:
    stamps = []
    for name in ("data_overview.json", "multimodal_data.json", "non_numerical_data.txt", "hierarchical_deviation_map.json"):
        try:
            stat = (directory / name).stat()
            stamps.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamps.append((name, 0, 0))
    relevant = config.model_dump(
        mode="json", include={"connection", "models", "engine", "decision", "token_budget", "local", "instructions"}
    )
    # A refreshed catalog can change a role's context window, and with it the budget.
    try:
        from ..config.settings import catalog_cache_path

        stamps.append(("catalog", catalog_cache_path().stat().st_mtime_ns, 0))
    except OSError:
        stamps.append(("catalog", 0, 0))
    return json.dumps([str(directory), stamps, relevant, task_json], sort_keys=True, default=str)


def measure_participant(
    config: DashboardConfig,
    participant_dir: Path,
    task_spec: Any = None,
    *,
    settings: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    The engine's offline measurement of one record under this configuration:
    input tokens, Predictor budget, route, and for a decision model the question
    load per request. None when the record cannot be loaded.
    """
    from ..utils.core.route_preview import measure_record
    from .engine_bridge import normalise_instructions

    directory = Path(participant_dir)
    if not (directory / "data_overview.json").exists():
        return None
    task_json = task_spec.model_dump_json() if task_spec is not None and hasattr(task_spec, "model_dump_json") else ""
    key = _measure_key(config, directory, task_json)
    with _MEASURE_LOCK:
        if key in _MEASURE_CACHE:
            _MEASURE_CACHE.move_to_end(key)
            return _MEASURE_CACHE[key]
    try:
        result: Optional[Dict[str, Any]] = measure_record(
            directory,
            settings=settings if settings is not None else engine_settings(config),
            prediction_task_spec=task_spec,
            agent_instructions=normalise_instructions(config.instructions),
        )
    except Exception:
        result = None
    with _MEASURE_LOCK:
        _MEASURE_CACHE[key] = result
        while len(_MEASURE_CACHE) > _MEASURE_CACHE_SIZE:
            _MEASURE_CACHE.popitem(last=False)
    return result


def resolve_route(
    config: DashboardConfig,
    *,
    input_tokens: int,
    predictor_kind: str,
    route: str = "auto",
    prices: Optional[Dict[str, Dict[str, Any]]] = None,
    measurement: Optional[Dict[str, Any]] = None,
    settings: Any = None,
) -> Dict[str, Any]:
    """
    The evidence route a run will take, decided the way the engine decides it.

    An explicit `route` is taken as given. With an engine `measurement` of the
    record (`measure_participant`) the engine's own route decision stands.
    Otherwise the configured mode decides: `always` orchestrates, `never` goes
    direct, and `auto` goes direct when the record fits the Predictor input
    budget (or the lower threshold the configuration sets).
    """
    if measurement is not None and measurement.get("predictor_kind") == predictor_kind:
        budget = int(measurement.get("budget_tokens") or 0)
        threshold = int(measurement.get("threshold_tokens") or budget)
        if route in ROUTES:
            chosen, reason = route, "Route requested for this estimate."
        else:
            chosen, reason = str(measurement["route"]), str(measurement.get("route_reason") or "")
        return {
            "route": chosen,
            "mode": config.engine.orchestration_mode,
            "budget_tokens": budget,
            "threshold_tokens": threshold,
            "reason": reason,
            "measured": True,
            "predictor_input_tokens": int(measurement.get("input_tokens") or 0),
        }
    budget = predictor_input_budget(config, predictor_kind, prices, settings=settings)
    mode = config.engine.orchestration_mode
    threshold = budget
    override = int(config.engine.orchestration_threshold_tokens or 0)
    if override > 0:
        threshold = min(override, budget) if budget > 0 else override
    if route in ROUTES:
        chosen, reason = route, "Route requested for this estimate."
    elif mode == "always":
        chosen, reason = "orchestrated", "Orchestration is set to run on every attempt."
    elif mode == "never":
        chosen, reason = "direct", "Orchestration is disabled."
    elif input_tokens <= threshold:
        chosen = "direct"
        reason = f"The record ({input_tokens:,} tokens) fits the Predictor input ({threshold:,}); no orchestration."
    else:
        chosen = "orchestrated"
        reason = f"The record ({input_tokens:,} tokens) exceeds the Predictor input ({threshold:,}); orchestrated."
    return {
        "route": chosen,
        "mode": mode,
        "budget_tokens": budget,
        "threshold_tokens": threshold,
        "reason": reason,
        "measured": False,
        "predictor_input_tokens": int(input_tokens),
    }


def estimate_run(
    *,
    config: DashboardConfig,
    input_tokens: int,
    iterations: Optional[int] = None,
    plan_steps: int = DEFAULT_PLAN_STEPS,
    include_deep_report: bool = True,
    chunking_expected: bool = False,
    route: str = "auto",
    predictor_kind: Optional[str] = None,
    measurement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Projected tokens and USD for one participant, broken down per agent.

    `route` is "direct", "orchestrated", or "auto" to decide it as the engine
    would; `predictor_kind` ("llm" or "decision") defaults to what the
    Predictor role is set to. The result says which route it assumed.

    `input_tokens` is the record's overview size, which the fitted profiles of
    the orchestrated roles scale with. `measurement` is the engine's offline
    measurement of the same record (`measure_participant`): with it the route,
    the Predictor budget, the direct Predictor prompt and a decision model's
    state and question load are the engine's numbers.
    """
    iterations = int(iterations or config.engine.max_iterations)
    prices = pricing_index()
    kind = predictor_kind if predictor_kind in ("llm", "decision") else resolve_predictor_kind(config)
    if measurement is not None and measurement.get("predictor_kind") != kind:
        measurement = None
    routing = resolve_route(
        config, input_tokens=input_tokens, predictor_kind=kind, route=route, prices=prices, measurement=measurement
    )
    direct = routing["route"] == "direct"
    lines: List[Dict[str, Any]] = []
    total_prompt = 0
    total_completion = 0
    total_usd = 0.0
    priced_all = True

    # Roles that read every tool output grow with the size of the plan. A
    # direct attempt has no plan, so nothing scales with one.
    plan_scale = max(0.5, plan_steps / DEFAULT_PLAN_STEPS)

    def add_line(role: str, model_id: str, calls: int, prompt: int, completion: int) -> None:
        nonlocal total_prompt, total_completion, total_usd, priced_all
        price = _price(prices, model_id)
        usd: Optional[float] = None
        if price["prompt"] is not None and price["completion"] is not None:
            usd = (prompt / 1_000_000) * price["prompt"] + (completion / 1_000_000) * price["completion"]
            total_usd += usd
        else:
            priced_all = False
        total_prompt += prompt
        total_completion += completion
        lines.append(
            {
                "role": role,
                "model": model_id,
                "calls": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "usd": round(usd, 6) if usd is not None else None,
                "price_known": price["known"],
                "prompt_usd_per_mtok": price["prompt"],
                "completion_usd_per_mtok": price["completion"],
            }
        )

    def add(profile: AgentProfile, role_key: str, calls: float, repeat: int) -> None:
        if calls <= 0:
            return
        scale = plan_scale if profile.scales_with_plan and not direct else 1.0
        prompt = int((profile.prompt_base + profile.prompt_per_input * input_tokens * scale) * calls * repeat)
        completion = int(
            (profile.completion_base + profile.completion_per_input * input_tokens) * calls * repeat
        )
        add_line(profile.role, _role_model(config, role_key), int(calls * repeat), prompt, completion)

    if not direct:
        add(AGENT_PROFILES["orchestrator"], "orchestrator", 1, iterations)
        add(AGENT_PROFILES["tool"], "tool", plan_steps, iterations)
        if chunking_expected:
            add(AGENT_PROFILES["integrator"], "integrator", 2, iterations)

    predictor = AGENT_PROFILES["predictor"]
    if kind == "decision":
        # Input only: the state (the record, or the orchestrated evidence, packed
        # to the state budget) plus the questions, for each request.
        evidence = (
            input_tokens
            if direct
            else int(predictor.prompt_base + predictor.prompt_per_input * input_tokens * plan_scale)
        )
        if measurement is not None:
            # The engine's numbers: the state packed to the budget, the measured
            # question load of each round-one request, and a zoomed round with
            # the same load when a continuous output is refined.
            measured_state = int(measurement.get("input_tokens") or 0)
            evidence = measured_state if direct else evidence
            state = min(evidence, routing["budget_tokens"]) if routing["budget_tokens"] else evidence
            per_request = [int(v) for v in (measurement.get("question_tokens_per_request") or [])] or [
                DECISION_QUESTION_TOKENS
            ]
            if measurement.get("round_two_expected"):
                per_request = per_request + per_request
            requests = len(per_request) * iterations
            prompt = sum(state + q for q in per_request) * iterations
        else:
            state = min(evidence, routing["budget_tokens"]) if routing["budget_tokens"] else evidence
            requests = DECISION_REQUESTS_PER_ATTEMPT * iterations
            prompt = requests * (state + DECISION_QUESTION_TOKENS)
        add_line("predictor", _role_model(config, "predictor"), requests, int(prompt), 0)
        compiler = str(config.decision.compiler_model or "").strip() or _role_model(config, "orchestrator")
        book = QUESTION_BOOK_PROFILE
        add_line(book.role, compiler, 1, book.prompt_base, book.completion_base)
        # The decision critic reads the answers' stability and calls no model.
    else:
        if direct:
            # The Predictor reads the complete record plus its own instructions:
            # the engine's measured prompt when there is one.
            prompt_per_attempt = (
                int(measurement.get("input_tokens") or 0)
                if measurement is not None
                else predictor.prompt_base + input_tokens
            )
            add_line(
                "predictor",
                _role_model(config, "predictor"),
                iterations,
                int(prompt_per_attempt * iterations),
                int(predictor.completion_base * iterations),
            )
        else:
            add(predictor, "predictor", 1, iterations)
        add(AGENT_PROFILES["critic"], "critic", 1, iterations)
    if include_deep_report:
        add(COMMUNICATOR_PROFILE, "communicator", 1, 1)

    total_tokens = total_prompt + total_completion
    return {
        "input_tokens": input_tokens,
        "iterations": iterations,
        "plan_steps": 0 if direct else plan_steps,
        "include_deep_report": include_deep_report,
        "route": routing["route"],
        "route_reason": routing["reason"],
        "predictor_kind": kind,
        "predictor_budget_tokens": routing["budget_tokens"],
        "route_threshold_tokens": routing["threshold_tokens"],
        "predictor_input_tokens": routing["predictor_input_tokens"],
        "measured": bool(routing.get("measured")),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "usd": round(total_usd, 6) if priced_all else None,
        "usd_low": round(total_usd * (1 - ESTIMATE_SPREAD), 6) if priced_all else None,
        "usd_high": round(total_usd * (1 + ESTIMATE_SPREAD), 6) if priced_all else None,
        "spread": ESTIMATE_SPREAD,
        "fully_priced": priced_all,
        "lines": lines,
        "budget_utilisation": (
            round(total_tokens / config.token_budget.total_budget, 4)
            if config.token_budget.total_budget
            else None
        ),
    }


def actual_cost_from_usage(
    usage_by_model: Dict[str, Dict[str, int]], *, cached_only: bool = False
) -> Dict[str, Any]:
    """
    Exact spend from observed per-model prompt/completion counts.

    ``usage_by_model`` maps a model id to ``{"prompt": int, "completion": int}``.
    Set ``cached_only`` on a hot path: it reports the spend as unpriced rather
    than blocking on a catalog fetch.
    """
    prices = pricing_index(cached_only=cached_only)
    lines: List[Dict[str, Any]] = []
    total_usd = 0.0
    total_prompt = 0
    total_completion = 0
    fully_priced = True

    for model_id, counts in sorted(usage_by_model.items()):
        prompt = int(counts.get("prompt") or 0)
        completion = int(counts.get("completion") or 0)
        total_prompt += prompt
        total_completion += completion
        price = _price(prices, model_id)
        usd: Optional[float] = None
        if price["prompt"] is not None and price["completion"] is not None:
            usd = (prompt / 1_000_000) * price["prompt"] + (completion / 1_000_000) * price["completion"]
            total_usd += usd
        else:
            fully_priced = False
        lines.append(
            {
                "model": model_id,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "usd": round(usd, 6) if usd is not None else None,
                "prompt_usd_per_mtok": price["prompt"],
                "completion_usd_per_mtok": price["completion"],
            }
        )

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "usd": round(total_usd, 6) if fully_priced else None,
        "fully_priced": fully_priced,
        "lines": lines,
    }


def model_profile() -> Dict[str, Any]:
    """The coefficients behind the projection, so the number can be audited."""
    return {
        "default_plan_steps": DEFAULT_PLAN_STEPS,
        "steps_base": STEPS_BASE,
        "steps_per_domain": STEPS_PER_DOMAIN,
        "spread": ESTIMATE_SPREAD,
        "calibration": (
            "Linear fit against the token ledgers of completed runs. Roles that read every "
            "tool output also scale with plan size, which is itself derived from the number "
            "of available domains."
        ),
        "agents": {
            key: {
                "calls_per_iteration": p.calls_per_iteration,
                "prompt_base": p.prompt_base,
                "prompt_per_input_token": p.prompt_per_input,
                "completion_base": p.completion_base,
            }
            for key, p in {
                **AGENT_PROFILES,
                "communicator": COMMUNICATOR_PROFILE,
                "question_book": QUESTION_BOOK_PROFILE,
            }.items()
        },
        "decision": {
            "requests_per_attempt": DECISION_REQUESTS_PER_ATTEMPT,
            "question_tokens_per_request": DECISION_QUESTION_TOKENS,
            "state_question_reserve": DECISION_STATE_QUESTION_RESERVE,
            "tokenizer_ratio": DECISION_TOKENIZER_RATIO,
        },
        "routing": (
            "The engine measures each record offline with the run's settings (the same "
            "DataLoader, direct executor output and Predictor measurement a run performs) "
            "and decides the route as the run will. A record that fits is projected on the "
            "direct route: no Orchestrator, no tool steps, and the measured Predictor prompt. "
            "A larger one is projected on the orchestrated route."
        ),
    }
