"""
Cost projection for a COMPASS run.

The per-agent token model below is a linear fit against the token ledgers of
completed runs (``performance_report_*.json``), so the projection tracks the
size of the participant's own evidence rather than a flat guess. Every
coefficient is exposed through :func:`model_profile` so the estimate can be
inspected instead of trusted blindly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import pricing_index
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
    return {
        "prompt": row.get("prompt_usd_per_mtok"),
        "completion": row.get("completion_usd_per_mtok"),
        "context_length": row.get("context_length"),
        "known": bool(row),
    }


def _role_model(config: DashboardConfig, role: str) -> str:
    explicit = str(getattr(config.models.role_models, role, "") or "").strip()
    return explicit or config.models.default_model


def estimate_run(
    *,
    config: DashboardConfig,
    input_tokens: int,
    iterations: Optional[int] = None,
    plan_steps: int = DEFAULT_PLAN_STEPS,
    include_deep_report: bool = True,
    chunking_expected: bool = False,
) -> Dict[str, Any]:
    """Projected tokens and USD for one participant, broken down per agent."""
    iterations = int(iterations or config.engine.max_iterations)
    prices = pricing_index()
    lines: List[Dict[str, Any]] = []
    total_prompt = 0
    total_completion = 0
    total_usd = 0.0
    priced_all = True

    # Roles that read every tool output grow with the size of the plan.
    plan_scale = max(0.5, plan_steps / DEFAULT_PLAN_STEPS)

    def add(profile: AgentProfile, role_key: str, calls: float, repeat: int) -> None:
        nonlocal total_prompt, total_completion, total_usd, priced_all
        if calls <= 0:
            return
        scale = plan_scale if profile.scales_with_plan else 1.0
        prompt = int((profile.prompt_base + profile.prompt_per_input * input_tokens * scale) * calls * repeat)
        completion = int(
            (profile.completion_base + profile.completion_per_input * input_tokens) * calls * repeat
        )
        model_id = _role_model(config, role_key)
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
                "role": profile.role,
                "model": model_id,
                "calls": int(calls * repeat),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "usd": round(usd, 6) if usd is not None else None,
                "price_known": price["known"],
                "prompt_usd_per_mtok": price["prompt"],
                "completion_usd_per_mtok": price["completion"],
            }
        )

    add(AGENT_PROFILES["orchestrator"], "orchestrator", 1, iterations)
    add(AGENT_PROFILES["tool"], "tool", plan_steps, iterations)
    if chunking_expected:
        add(AGENT_PROFILES["integrator"], "integrator", 2, iterations)
    add(AGENT_PROFILES["predictor"], "predictor", 1, iterations)
    add(AGENT_PROFILES["critic"], "critic", 1, iterations)
    if include_deep_report:
        add(COMMUNICATOR_PROFILE, "communicator", 1, 1)

    total_tokens = total_prompt + total_completion
    return {
        "input_tokens": input_tokens,
        "iterations": iterations,
        "plan_steps": plan_steps,
        "include_deep_report": include_deep_report,
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
            for key, p in {**AGENT_PROFILES, "communicator": COMMUNICATOR_PROFILE}.items()
        },
    }
