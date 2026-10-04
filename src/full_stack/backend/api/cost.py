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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..decision import get_decision_model_spec
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

#: The engine's fallback when nothing reports the Predictor's context window.
FALLBACK_CONTEXT_TOKENS = 128_000
ROUTES = ("direct", "orchestrated")


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


def predictor_input_budget(
    config: DashboardConfig,
    predictor_kind: str,
    prices: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """
    The input the Predictor accepts, in the cl100k tokens the engine counts.

    Mirrors the engine: for an LLM, the context window minus the output
    reserve, capped by the agent input limit; for a decision model, its state
    limit converted to cl100k tokens minus what the longest question needs.
    """
    model_id = _role_model(config, "predictor")
    if predictor_kind == "decision":
        spec = get_decision_model_spec(model_id)
        if spec is None:
            return 0
        state = spec.state_budget_cl100k(ratio=DECISION_TOKENIZER_RATIO)
        return max(0, state - DECISION_STATE_QUESTION_RESERVE)

    if config.connection.backend == "local":
        context = int(config.local.max_model_len or config.local.max_tokens)
        default_context = context
    else:
        row = _price(prices or {}, model_id)
        explicit = int(config.models.context_window or 0)
        default_context = explicit or int(
            _price(prices or {}, config.models.default_model)["context_length"] or FALLBACK_CONTEXT_TOKENS
        )
        if explicit and model_id == config.models.default_model:
            context = explicit
        else:
            context = int(row["context_length"] or default_context)

    completion = int(config.models.role_max_tokens.predictor or 0) or 4096
    reserve = max(1024, min(8192, completion + 1024))
    by_context = max(2048, context - reserve)
    cap = int(config.token_budget.max_agent_input_tokens or 0)
    if not cap:
        # The engine derives the agent input limit from the default model's window.
        if config.connection.backend == "local":
            cap = max(4096, min(int(default_context * 0.75), default_context - 16_000 - 2048))
        else:
            cap = int(default_context * 0.95)
    return max(2048, min(cap, by_context))


def resolve_route(
    config: DashboardConfig,
    *,
    input_tokens: int,
    predictor_kind: str,
    route: str = "auto",
    prices: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    The evidence route a run will take, decided the way the engine decides it.

    An explicit `route` is taken as given. Otherwise the configured mode
    decides: `always` orchestrates, `never` goes direct, and `auto` goes direct
    when the record fits the Predictor input budget (or the lower threshold the
    configuration sets).
    """
    budget = predictor_input_budget(config, predictor_kind, prices)
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
    return {"route": chosen, "mode": mode, "budget_tokens": budget, "threshold_tokens": threshold, "reason": reason}


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
) -> Dict[str, Any]:
    """
    Projected tokens and USD for one participant, broken down per agent.

    `route` is "direct", "orchestrated", or "auto" to decide it as the engine
    would; `predictor_kind` ("llm" or "decision") defaults to what the
    Predictor role is set to. The result says which route it assumed.
    """
    iterations = int(iterations or config.engine.max_iterations)
    prices = pricing_index()
    kind = predictor_kind if predictor_kind in ("llm", "decision") else resolve_predictor_kind(config)
    routing = resolve_route(
        config, input_tokens=input_tokens, predictor_kind=kind, route=route, prices=prices
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
        state = min(evidence, routing["budget_tokens"]) if routing["budget_tokens"] else evidence
        requests = DECISION_REQUESTS_PER_ATTEMPT * iterations
        add_line(
            "predictor",
            _role_model(config, "predictor"),
            requests,
            int(requests * (state + DECISION_QUESTION_TOKENS)),
            0,
        )
        compiler = str(config.decision.compiler_model or "").strip() or _role_model(config, "orchestrator")
        book = QUESTION_BOOK_PROFILE
        add_line(book.role, compiler, 1, book.prompt_base, book.completion_base)
        # The decision critic reads the answers' stability and calls no model.
    else:
        if direct:
            # The Predictor reads the complete record plus its own instructions.
            add_line(
                "predictor",
                _role_model(config, "predictor"),
                iterations,
                int((predictor.prompt_base + input_tokens) * iterations),
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
            "A record that fits the Predictor input is projected on the direct route: no "
            "Orchestrator, no tool steps, and a Predictor prompt of the record plus its "
            "instructions. A larger one is projected on the orchestrated route."
        ),
    }
