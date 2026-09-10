"""Service identity, engine capabilities, and health."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Path as PathParam, Query

from ...config.settings import COMPASS_FULL_NAME, COMPASS_VERSION
from ...data.models.execution_plan import ToolName
from ...runtime.event_bus import STAGE_NAMES
from .. import DEFAULT_MODEL
from ..catalog import connectivity
from ..cost import model_profile
from ..flow import TOOL_FAMILIES
from ..schemas import AGENT_ROLES, INSTRUCTION_SLOTS
from ..storage import clear as clear_storage
from ..storage import describe as describe_storage

router = APIRouter(tags=["system"])

TOOL_SUMMARIES: Dict[str, str] = {
    "UnimodalCompressor": "Compresses one domain into a dense, citable summary.",
    "MultimodalNarrativeCreator": "Weaves several domains into a single narrative.",
    "HypothesisGenerator": "Proposes candidate mechanisms from the evidence.",
    "CodeExecutor": "Runs deterministic computation over the feature values.",
    "FeatureSynthesizer": "Derives higher-order features from raw measurements.",
    "ClinicalRelevanceRanker": "Ranks findings by relevance to the target.",
    "AnomalyNarrativeBuilder": "Describes the strongest deviations in context.",
    "PhenotypeRepresentation": "Builds a structured phenotype representation.",
    "DifferentialDiagnosis": "Contrasts the target against plausible alternatives.",
}

#: `has_model` marks the roles that carry their own model setting. The Executor
#: is an agent in the workflow but dispatches tools, so it uses the tool model
#: rather than one of its own: consumers must not index a per-role map by it.
AGENT_SUMMARIES = [
    {"role": "orchestrator", "label": "Orchestrator", "stage": 1, "has_model": True,
     "summary": "Reads the coverage map and writes an execution plan of tool steps with dependencies."},
    {"role": "executor", "label": "Executor", "stage": 2, "has_model": False,
     "summary": "Runs unblocked plan steps concurrently and repairs failures in place. Uses the tool model."},
    {"role": "integrator", "label": "Integrator", "stage": 3, "has_model": True,
     "summary": "Fuses step outputs, tracks feature coverage, and chunks evidence when the payload overflows."},
    {"role": "predictor", "label": "Predictor", "stage": 4, "has_model": True,
     "summary": "Produces the phenotype outputs with an explicit evidence chain."},
    {"role": "critic", "label": "Critic", "stage": 5, "has_model": True,
     "summary": "Scores the prediction against a task-specific checklist and decides whether to iterate."},
    {"role": "communicator", "label": "Communicator", "stage": 6, "has_model": True,
     "summary": "Writes the evidence-grounded deep phenotype report and marks missing information."},
]


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": COMPASS_VERSION}


@router.get("/system/connectivity")
def system_connectivity(refresh: bool = Query(False, description="Skip the cached probe")) -> Dict[str, Any]:
    """
    Whether this machine can reach the network and the configured provider.

    A healthy connection is shown by showing nothing at all, so this exists to
    make the failing case visible and is cached to stay cheap to poll.
    """
    return connectivity(force=refresh)


@router.get("/system/storage")
def system_storage() -> Dict[str, Any]:
    """Everything COMPASS has written on this machine, with its size on disk."""
    return describe_storage()


@router.delete("/system/storage/{key}")
def delete_system_storage(key: str = PathParam(..., description="One storage entry key")) -> Dict[str, Any]:
    """
    Remove one entry's contents.

    Only the fixed set of keys above is accepted, so this can never be aimed at
    a path of the caller's choosing.
    """
    try:
        return clear_storage(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"No such storage entry: {key}") from exc


@router.get("/capabilities")
def capabilities() -> Dict[str, Any]:
    """Everything the interface needs to describe the engine without hardcoding it."""
    return {
        "name": "COMPASS",
        "full_name": COMPASS_FULL_NAME,
        "version": COMPASS_VERSION,
        "default_model": DEFAULT_MODEL,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "stages": STAGE_NAMES,
        "agent_roles": list(AGENT_ROLES),
        "instruction_slots": list(INSTRUCTION_SLOTS),
        "agents": AGENT_SUMMARIES,
        "tools": [
            {
                "name": tool.value,
                "family": TOOL_FAMILIES.get(tool.value, "other"),
                "summary": TOOL_SUMMARIES.get(tool.value, ""),
            }
            for tool in ToolName
        ],
        "prediction_types": [
            {"value": "binary", "label": "Binary classification",
             "summary": "One target against one comparator.", "needs": ["target_label", "control_label"]},
            {"value": "multiclass", "label": "Multiclass classification",
             "summary": "Three or more mutually exclusive classes.", "needs": ["class_labels"]},
            {"value": "regression_univariate", "label": "Univariate regression",
             "summary": "A single continuous output.", "needs": ["regression_outputs"]},
            {"value": "regression_multivariate", "label": "Multivariate regression",
             "summary": "Two or more continuous outputs.", "needs": ["regression_outputs"]},
            {"value": "hierarchical", "label": "Hierarchical task tree",
             "summary": "Mixed modes arranged as a tree of dependent questions.", "needs": ["root"]},
        ],
        "cost_model": model_profile(),
        "provider_links": {
            "openrouter_keys": "https://openrouter.ai/settings/keys",
            "openrouter_models": "https://openrouter.ai/models",
            "openrouter_credits": "https://openrouter.ai/settings/credits",
            "openrouter_signup": "https://openrouter.ai/sign-up",
        },
    }
