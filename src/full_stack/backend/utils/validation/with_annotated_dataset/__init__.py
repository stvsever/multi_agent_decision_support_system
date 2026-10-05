"""Annotated dataset evaluation library for COMPASS.

Evaluate COMPASS predictions against an annotated dataset, whatever the input
and output structure: binary, multiclass, multi-label, hierarchical and mixed
trees, univariate and multivariate regression. See README.md.

Quick start::

    from src.full_stack.backend.utils.validation.with_annotated_dataset import evaluate
    result = evaluate("results/participant_runs", "annotations.json", out_dir="analysis")
"""

from .core import (
    SUPPORTED_PREDICTION_TYPES,
    TIDY_COLUMNS,
    EvaluationResult,
    MetricTables,
    TaskStructure,
    build_tidy,
    compare_configurations,
    compute_metrics,
    evaluate,
    load_annotations,
    load_predictions,
    load_task_spec,
    make_figures,
    metrics_wide,
    parse_disorder_groups,
    read_tidy,
    run_detailed_workflow,
    run_metrics_workflow,
    summarize_annotation_contract,
    test_retest,
    tidy_to_csv,
    write_outputs,
)

__all__ = [
    "SUPPORTED_PREDICTION_TYPES",
    "summarize_annotation_contract",
    "parse_disorder_groups",
    "run_metrics_workflow",
    "run_detailed_workflow",
    "TIDY_COLUMNS",
    "EvaluationResult",
    "MetricTables",
    "TaskStructure",
    "build_tidy",
    "compare_configurations",
    "compute_metrics",
    "evaluate",
    "load_annotations",
    "load_predictions",
    "load_task_spec",
    "make_figures",
    "metrics_wide",
    "read_tidy",
    "test_retest",
    "tidy_to_csv",
    "write_outputs",
]
