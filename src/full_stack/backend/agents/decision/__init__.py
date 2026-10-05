"""
Structured decision models in the Predictor role.

    registry   which model ids are decision models, their limits and prices
    client     HTTP transport (OpenRouter Decisions API or TypeSafe native)
    scales     regression outputs as ordered levels, density, continuous estimate
    questions  task specification to a frozen question book and typed questions
    state      the evidence a decision model reads, packed to its limit
    aggregate  answers to node predictions (order ensembles, distributions)
    predictor  the DecisionPredictor agent
    quality    the decision critic

Only the Predictor can be a decision model. Every other role needs a
conventional LLM; `enforce_role_models` checks that.
"""

from .registry import (
    DecisionModelSpec,
    get_decision_model_spec,
    is_decision_model,
    list_decision_models,
    register_decision_model,
)

NON_PREDICTOR_ROLES = ("orchestrator", "critic", "integrator", "communicator", "tool")


def decision_role_conflicts(role_models: dict) -> list:
    """Roles other than the Predictor that are set to a decision model."""
    return [role for role in NON_PREDICTOR_ROLES if is_decision_model((role_models or {}).get(role))]


def enforce_role_models(settings) -> None:
    """Raise when a role other than the Predictor is set to a decision model."""
    models = settings.models
    conflicts = decision_role_conflicts({role: getattr(models, f"{role}_model", "") for role in NON_PREDICTOR_ROLES})
    if conflicts:
        raise ValueError(
            "Structured decision models can only serve the Predictor role. "
            f"Set a conventional LLM for: {', '.join(conflicts)}."
        )


__all__ = [
    "DecisionModelSpec",
    "NON_PREDICTOR_ROLES",
    "decision_role_conflicts",
    "enforce_role_models",
    "get_decision_model_spec",
    "is_decision_model",
    "list_decision_models",
    "register_decision_model",
]
