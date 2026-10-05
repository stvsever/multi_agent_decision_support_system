"""
Structured decision model registry.

A structured decision model (a "System One" model such as TypeSafe Jev) does not
generate text. It reads a state, answers typed questions (Noul, Choice, Score)
and returns calibrated probabilities. In COMPASS such a model can only serve the
Predictor role; every other role needs a conventional LLM.

This module knows which model ids are decision models and what their limits and
prices are. It imports nothing from the engine, so the settings module can use
it without an import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DecisionModelSpec:
    """Capabilities and limits of one structured decision model."""

    model_id: str
    label: str
    # Limit for the state plus the single longest question, in provider tokens.
    state_context_tokens: int
    # Limit for the state plus all questions of one request, in provider tokens.
    request_context_tokens: int
    max_choice_options: int = 255
    max_score_levels: int = 10
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    # Id understood by the provider's native API (TypeSafe), when it differs.
    native_id: str = ""

    def state_budget_cl100k(self, *, ratio: float = 1.2) -> int:
        """State limit converted to the cl100k tokens the engine counts with."""
        safe_ratio = max(1.0, float(ratio or 1.0))
        return int(self.state_context_tokens / safe_ratio)

    def as_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "kind": "decision",
            "state_context_tokens": self.state_context_tokens,
            "request_context_tokens": self.request_context_tokens,
            "max_choice_options": self.max_choice_options,
            "max_score_levels": self.max_score_levels,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "roles": ["predictor"],
        }


_JEV_1_13 = DecisionModelSpec(
    model_id="typesafe/jev-1.13",
    label="TypeSafe Jev 1.13",
    state_context_tokens=32_000,
    request_context_tokens=64_000,
    input_price_per_million=0.042,
    output_price_per_million=0.0,
    native_id="jev-1.13",
)

_REGISTRY: Dict[str, DecisionModelSpec] = {
    "typesafe/jev-1.13": _JEV_1_13,
    "typesafe/jev-latest": DecisionModelSpec(
        model_id="typesafe/jev-latest",
        label="TypeSafe Jev (latest)",
        state_context_tokens=_JEV_1_13.state_context_tokens,
        request_context_tokens=_JEV_1_13.request_context_tokens,
        input_price_per_million=_JEV_1_13.input_price_per_million,
        output_price_per_million=0.0,
        native_id="jev-latest",
    ),
}

# Any other TypeSafe System One id ("typesafe/jev-1.14", "jev-preview") is
# treated as a Jev-class decision model with the 1.13 limits until registered.
# The Jev Router is a text model router, not a decision model.
_SYSTEM_ONE_PATTERN = re.compile(r"^(?:typesafe/)?jev-(?!router)[a-z0-9.\-]+$")


def normalize_model_id(model: Optional[str]) -> str:
    text = str(model or "").strip().lower()
    if text.startswith("~"):
        text = text[1:]
    return text


def get_decision_model_spec(model: Optional[str]) -> Optional[DecisionModelSpec]:
    """Return the spec when `model` is a structured decision model, else None."""
    key = normalize_model_id(model)
    if not key:
        return None
    if key in _REGISTRY:
        return _REGISTRY[key]
    bare = key[len("typesafe/"):] if key.startswith("typesafe/") else key
    if f"typesafe/{bare}" in _REGISTRY:
        return _REGISTRY[f"typesafe/{bare}"]
    if _SYSTEM_ONE_PATTERN.match(key):
        return DecisionModelSpec(
            model_id=f"typesafe/{bare}",
            label=f"TypeSafe {bare}",
            state_context_tokens=_JEV_1_13.state_context_tokens,
            request_context_tokens=_JEV_1_13.request_context_tokens,
            input_price_per_million=_JEV_1_13.input_price_per_million,
            output_price_per_million=0.0,
            native_id=bare,
        )
    return None


def is_decision_model(model: Optional[str]) -> bool:
    return get_decision_model_spec(model) is not None


def register_decision_model(spec: DecisionModelSpec) -> None:
    """Register another decision model (for example a newer Jev release)."""
    _REGISTRY[normalize_model_id(spec.model_id)] = spec


def list_decision_models() -> List[DecisionModelSpec]:
    return list(_REGISTRY.values())


def native_model_id(model: str) -> str:
    """Id for the provider's native API: `typesafe/jev-1.13` becomes `jev-1.13`."""
    spec = get_decision_model_spec(model)
    if spec is not None and spec.native_id:
        return spec.native_id
    key = normalize_model_id(model)
    return key.split("/", 1)[1] if "/" in key else key
