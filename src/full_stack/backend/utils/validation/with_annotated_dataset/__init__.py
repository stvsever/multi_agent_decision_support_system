"""Annotated dataset validation package for COMPASS.

This package provides mode-aware validation across binary, multiclass,
regression, and hierarchical prediction tasks.
"""

from .core import (
    SUPPORTED_PREDICTION_TYPES,
    parse_disorder_groups,
    run_detailed_workflow,
    run_metrics_workflow,
    summarize_annotation_contract,
)

__all__ = [
    "SUPPORTED_PREDICTION_TYPES",
    "summarize_annotation_contract",
    "parse_disorder_groups",
    "run_metrics_workflow",
    "run_detailed_workflow",
]
