"""Core modules for annotated-dataset validation workflows.

New evaluation API (tidy table in, metrics and figures out):
``evaluate``, ``build_tidy``, ``compute_metrics``, ``compare_configurations``,
``test_retest`` and the figure functions in ``core.figures``. The legacy
mode-specific workflows (``run_metrics_workflow``, ``run_detailed_workflow``)
are kept unchanged for the existing CLIs.
"""

from .annotation_contract import summarize_annotation_contract
from .constants import SUPPORTED_PREDICTION_TYPES
from .evaluate import EvaluationResult, evaluate, make_figures, write_outputs
from .stats import (
    MetricTables,
    compare_configurations,
    compute_metrics,
    metrics_wide,
    test_retest,
)
from .tidy import (
    TIDY_COLUMNS,
    TaskStructure,
    build_tidy,
    load_annotations,
    load_predictions,
    load_task_spec,
    read_tidy,
    tidy_to_csv,
)
from .workflows import parse_disorder_groups, run_detailed_workflow, run_metrics_workflow

__all__ = [
    "SUPPORTED_PREDICTION_TYPES",
    "summarize_annotation_contract",
    "parse_disorder_groups",
    "run_metrics_workflow",
    "run_detailed_workflow",
    "EvaluationResult",
    "evaluate",
    "make_figures",
    "write_outputs",
    "MetricTables",
    "compute_metrics",
    "compare_configurations",
    "metrics_wide",
    "test_retest",
    "TIDY_COLUMNS",
    "TaskStructure",
    "build_tidy",
    "load_annotations",
    "load_predictions",
    "load_task_spec",
    "read_tidy",
    "tidy_to_csv",
]
