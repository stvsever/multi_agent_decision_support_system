"""Single entry point: predictions + annotations -> tidy table, metrics, figures.

Python::

    from <package>.with_annotated_dataset import evaluate
    result = evaluate("results/", "annotations.json", group_by=["tier", "predictor"], out_dir="analysis/")
    result.metrics          # long metric table with 95% CIs
    result.metrics_wide     # one row per group x node x output

CLI::

    python -m <package>.with_annotated_dataset evaluate \\
        --predictions results/ --annotations annotations.json \\
        --group-by tier,predictor --out analysis/

(``python run_evaluation.py evaluate ...`` inside this folder works too.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .stats import (
    DEFAULT_CI,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    MetricTables,
    compare_configurations,
    compute_metrics,
    metrics_wide,
    test_retest,
)
from .tidy import TaskStructure, TIDY_COLUMNS, build_tidy, load_predictions, load_task_spec, read_tidy, tidy_to_csv

_TIDY_REQUIRED = {"participant_id", "node_id", "output", "kind", "status"}


@dataclass
class EvaluationResult:
    """Everything ``evaluate`` produced (tables are pandas DataFrames)."""

    tidy: pd.DataFrame
    structure: Optional[TaskStructure]
    metrics: pd.DataFrame
    metrics_wide: pd.DataFrame
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    multilabel: pd.DataFrame
    hierarchy: pd.DataFrame
    comparison: pd.DataFrame
    test_retest: pd.DataFrame
    summary: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)


def _slug(text: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text).strip())
    return re.sub(r"_+", "_", slug).strip("_")[:80] or "x"


def _is_tidy(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and _TIDY_REQUIRED.issubset(frame.columns)


def _tidy_file(path: Any) -> bool:
    if not isinstance(path, (str, Path)):
        return False
    p = Path(str(path))
    if p.suffix.lower() not in {".csv", ".tsv"} or not p.exists():
        return False
    header = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",", nrows=0).columns
    return _TIDY_REQUIRED.issubset(header)


def _write_table(frame: pd.DataFrame, out_dir: Path, name: str, files: Dict[str, str], json_too: bool = False) -> None:
    if frame is None or frame.empty:
        return
    csv_path = out_dir / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    files[name] = str(csv_path)
    if json_too:
        json_path = out_dir / f"{name}.json"
        json_path.write_text(frame.to_json(orient="records", indent=2, default_handler=str))
        files[f"{name}_json"] = str(json_path)


def _parse_compare(compare: Any, tidy: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Normalize ``compare`` into (column, a, b) triples."""
    if compare is None:
        return []
    # A string or a tuple is one comparison; a list holds several.
    items = list(compare) if isinstance(compare, list) else [compare]
    out: List[Tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, str):
            col, _, rest = item.partition(":")
            values = [v for v in rest.split(",") if v] if rest else []
        else:
            col, *values = list(item)
        if col not in tidy.columns:
            raise ValueError(f"compare column '{col}' is not in the tidy table")
        if not values:
            values = sorted(tidy[col].dropna().astype(str).unique())
        if len(values) < 2:
            continue
        a = str(values[0])
        for b in values[1:]:
            out.append((col, a, str(b)))
    return out


def evaluate(
    predictions: Any,
    annotations: Any = None,
    *,
    task_spec: Any = None,
    group_by: Sequence[str] = (),
    out_dir: Optional[Union[str, Path]] = None,
    compare: Any = None,
    repeat_col: Optional[str] = "auto",
    multilabel_groups: Optional[Mapping[str, Sequence[str]]] = None,
    positive_labels: Optional[Mapping[str, str]] = None,
    expected_participants: str = "observed",
    id_column: Optional[str] = None,
    annotation_columns: Optional[Mapping[str, str]] = None,
    annotation_targets: Optional[Sequence[str]] = None,
    prediction_columns: Optional[Mapping[str, str]] = None,
    path_groups: Sequence[str] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    figures: bool = True,
    figure_formats: Sequence[str] = ("png",),
    tier_col: str = "tier",
    tier_order: Optional[Sequence[str]] = None,
    max_facets: int = 12,
) -> EvaluationResult:
    """Evaluate COMPASS predictions against annotations.

    Parameters
    ----------
    predictions:
        COMPASS result folder(s), JSON/JSONL job records, CSV/TSV, a DataFrame,
        dicts, or an already tidy DataFrame / tidy CSV (used as is).
    annotations:
        JSON template, JSONL, CSV/TSV, DataFrame or dict; None when the truth
        is embedded in the records (``ground_truth`` / ``labels_truth``).
    task_spec:
        PredictionTaskSpec object, dict or JSON file; inferred from COMPASS
        reports or from the data when absent.
    group_by:
        Columns to stratify every metric by (for example ``["tier", "predictor"]``).
    compare:
        Paired comparison(s): ``("predictor", "llm", "decision")``, ``"predictor:llm,decision"``
        or a list of these; ``"predictor"`` alone compares the first value with each other.
    repeat_col:
        Column holding the repeat index for test-retest agreement; ``"auto"``
        uses ``repeat`` when present with two or more values; None disables.
    """
    group_by = [g for g in (group_by or []) if g]
    if _is_tidy(predictions):
        tidy = predictions.copy()
        structure = load_task_spec(task_spec, positive_labels) if task_spec is not None else None
    elif _tidy_file(predictions):
        tidy = read_tidy(predictions)
        structure = load_task_spec(task_spec, positive_labels) if task_spec is not None else None
    else:
        pred_kwargs = {"id_column": id_column, "path_groups": tuple(path_groups),
                       "prediction_columns": prediction_columns, "group_columns": tuple(group_by)}
        ann_kwargs = {"id_column": id_column, "columns": annotation_columns, "targets": annotation_targets}
        tidy, structure = build_tidy(
            predictions,
            annotations,
            task_spec=task_spec,
            group_by=group_by,
            expected_participants=expected_participants,
            positive_labels=positive_labels,
            prediction_kwargs=pred_kwargs,
            annotation_kwargs=ann_kwargs,
            return_structure=True,
        )
    missing = [g for g in group_by if g not in tidy.columns]
    if missing:
        available = [c for c in tidy.columns if c not in TIDY_COLUMNS]
        raise ValueError(f"group_by columns not found: {missing}. Available grouping columns: {available}")

    tables: MetricTables = compute_metrics(tidy, group_by, structure=structure, n_boot=n_boot, seed=seed, ci=ci,
                                           multilabel_groups=multilabel_groups)
    comparisons = []
    for col, a, b in _parse_compare(compare, tidy):
        comparisons.append(compare_configurations(tidy, col, a, b, [g for g in group_by if g != col],
                                                  n_boot=n_boot, seed=seed, ci=ci))
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()

    retest = pd.DataFrame()
    rcol = repeat_col
    if rcol == "auto":
        rcol = "repeat" if "repeat" in tidy.columns and tidy["repeat"].nunique(dropna=True) >= 2 else None
    if rcol and rcol in tidy.columns and rcol not in group_by:
        retest = test_retest(tidy, rcol, group_by, n_boot=n_boot, seed=seed, ci=ci)

    status_counts = tidy["status"].value_counts().to_dict()
    summary: Dict[str, Any] = {
        "n_rows": int(len(tidy)),
        "n_participants": int(tidy["participant_id"].nunique()),
        "n_instances": int(tidy["instance_id"].nunique()) if "instance_id" in tidy.columns else None,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "kinds": sorted(tidy["kind"].dropna().unique().tolist()),
        "group_by": group_by,
        "compare": [list(c) for c in _parse_compare(compare, tidy)],
        "repeat_col": rcol,
        "bootstrap": tables.settings,
        "task_structure": structure.to_dict() if structure is not None else None,
        "recalibration": "none (zero-shot: raw predictions are scored)",
    }
    result = EvaluationResult(
        tidy=tidy,
        structure=structure,
        metrics=tables.metrics,
        metrics_wide=metrics_wide(tables.metrics),
        per_class=tables.per_class,
        confusion=tables.confusion,
        multilabel=tables.multilabel,
        hierarchy=tables.hierarchy,
        comparison=comparison,
        test_retest=retest,
        summary=summary,
    )
    if out_dir is not None:
        write_outputs(result, out_dir, figures=figures, figure_formats=figure_formats, tier_col=tier_col,
                      tier_order=tier_order, max_facets=max_facets)
    return result


def write_outputs(
    result: EvaluationResult,
    out_dir: Union[str, Path],
    *,
    figures: bool = True,
    figure_formats: Sequence[str] = ("png",),
    tier_col: str = "tier",
    tier_order: Optional[Sequence[str]] = None,
    max_facets: int = 12,
) -> EvaluationResult:
    """Write tables (CSV, metrics also JSON), ``summary.json`` and figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    tidy_path = out / "tidy.csv"
    tidy_to_csv(result.tidy, tidy_path)
    files["tidy"] = str(tidy_path)
    _write_table(result.metrics, out, "metrics_long", files, json_too=True)
    _write_table(result.metrics_wide, out, "metrics_wide", files, json_too=True)
    _write_table(result.per_class, out, "per_class", files)
    _write_table(result.confusion, out, "confusion", files)
    _write_table(result.multilabel, out, "multilabel", files, json_too=True)
    _write_table(result.hierarchy, out, "hierarchy", files, json_too=True)
    _write_table(result.comparison, out, "comparison", files, json_too=True)
    _write_table(result.test_retest, out, "test_retest", files, json_too=True)
    if figures:
        result.figures = make_figures(result, out / "figures", formats=figure_formats, tier_col=tier_col,
                                      tier_order=tier_order, max_facets=max_facets)
    result.files = files
    summary = dict(result.summary)
    summary["files"] = files
    summary["figures"] = result.figures
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    files["summary"] = str(out / "summary.json")
    return result


def make_figures(
    result: EvaluationResult,
    fig_dir: Union[str, Path],
    *,
    formats: Sequence[str] = ("png",),
    tier_col: str = "tier",
    tier_order: Optional[Sequence[str]] = None,
    max_facets: int = 12,
) -> List[str]:
    """The standard figure set for one evaluation (returns written paths)."""
    from . import figures as F

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tidy, metrics = result.tidy, result.metrics
    group_by = list(result.summary.get("group_by") or [])
    written: List[str] = []
    if metrics.empty:
        metrics = pd.DataFrame(columns=["node_id", "output", "kind", "metric", "value", "ci_low", "ci_high"])
    color_col = group_by[-1] if len(group_by) >= 2 else None
    label_cols = [*group_by, "output"]
    # Fix each configuration's color once so subsets never repaint it.
    config_cols = [g for g in group_by if g != tier_col and g in tidy.columns]
    F.register_colors(tidy[config_cols].astype(str).agg(" | ".join, axis=1).unique().tolist() if config_cols else [])

    def _save(fig: Any, name: str) -> None:
        written.extend(F.save_figure(fig, fig_dir / name, formats))

    # Regression outputs: density scatter grid and residual diagnostics.
    reg = tidy[tidy["kind"] == "regression"]
    for (node_id, output), _ in reg.groupby(["node_id", "output"], sort=False):
        fig = F.scatter_grid(tidy, node_id, output, facet_cols=group_by, metrics=metrics, max_facets=max_facets)
        _save(fig, f"scatter_{_slug(node_id)}__{_slug(output)}")
        fig = F.residual_plot(tidy, node_id, output, color_col=group_by[0] if group_by else None)
        _save(fig, f"residuals_{_slug(node_id)}__{_slug(output)}")
    # Forest plots of the headline metrics.
    for metric in ("pearson_r", "nmae", "balanced_accuracy", "auroc"):
        sub = metrics[(metrics["metric"] == metric) & np.isfinite(metrics["value"].astype(float))]
        if sub.empty:
            continue
        fig = F.forest_plot(sub, metric, label_cols=label_cols, color_col=color_col)
        _save(fig, f"forest_{metric}")
    # Metric by tier, one line per configuration.
    if tier_col in group_by:
        config_cols = [g for g in group_by if g != tier_col]
        for (node_id, output, kind), _ in metrics.groupby(["node_id", "output", "kind"], sort=False):
            if output == "__macro__":
                continue
            for metric in (("pearson_r", "mae") if kind == "regression" else ("balanced_accuracy", "auroc")):
                sub = metrics[(metrics["node_id"] == node_id) & (metrics["output"] == output) & (metrics["metric"] == metric)]
                if sub.empty or not np.isfinite(sub["value"].astype(float)).any():
                    continue
                fig = F.metric_by_tier(sub, metric, tier_col=tier_col, config_cols=config_cols, tier_order=tier_order,
                                       title=f"{output}: {F.metric_label(metric)} by {tier_col} (95% CI)")
                _save(fig, f"by_{_slug(tier_col)}_{_slug(output)}__{metric}")
    # Confusion matrices and ROC curves.
    if not result.confusion.empty:
        gcols = [g for g in group_by if g in result.confusion.columns]
        for node_id, cframe in result.confusion.groupby("node_id", sort=False):
            groups = list(cframe.groupby(gcols, dropna=False, sort=True)) if gcols else [((), cframe)]
            for key, sub in groups[:max_facets]:
                key = key if isinstance(key, tuple) else (key,)
                suffix = "__".join(_slug(k) for k in key)
                title = f"{node_id}" + (f" ({' | '.join(str(k) for k in key)})" if key else "")
                fig = F.confusion_matrix_plot(sub, title=f"Confusion matrix: {title}")
                _save(fig, f"confusion_{_slug(node_id)}" + (f"__{suffix}" if suffix else ""))
        cls = tidy[tidy["kind"].isin(["binary", "multiclass"]) & (tidy["status"] == "ok")]
        for node_id, nframe in cls.groupby("node_id", sort=False):
            has_prob = nframe["p_positive"].notna().any() if str(nframe["kind"].iloc[0]) == "binary" else nframe["probs"].apply(lambda p: isinstance(p, dict) and bool(p)).any()
            if not has_prob:
                continue
            fig = F.roc_plot(tidy, node_id, color_cols=group_by)
            _save(fig, f"roc_{_slug(node_id)}")
            if str(nframe["kind"].iloc[0]) == "binary":
                fig = F.calibration_plot(tidy, node_id, color_cols=group_by)
                _save(fig, f"calibration_{_slug(node_id)}")
    # Multi-label per label F1.
    if not result.multilabel.empty:
        ml = result.multilabel[(result.multilabel["metric"] == "f1") & (result.multilabel["output"] != "__all__")]
        if not ml.empty:
            fig = F.forest_plot(ml, "f1", label_cols=[*group_by, "node_id", "output"], color_col=color_col,
                                title="Multi-label F1 per label (95% CI)")
            _save(fig, "multilabel_f1")
    # Paired comparisons.
    if not result.comparison.empty:
        for (col, a, b), cframe in result.comparison.groupby(["config_col", "config_a", "config_b"], sort=False):
            other = [g for g in group_by if g != col]
            for metric in ("mae", "nmae", "pearson_r", "balanced_accuracy", "auroc"):
                sub = cframe[cframe["metric"] == metric]
                if sub.empty:
                    continue
                fig = F.paired_comparison_plot(sub, metric=metric, label_cols=[*other, "output"],
                                               title=f"{F.metric_label(metric)}: {b} vs {a} ({col}), matched participants")
                _save(fig, f"paired_{_slug(col)}_{_slug(b)}_vs_{_slug(a)}__{metric}")
            for (node_id, output), _ in cframe[cframe["kind"] == "regression"].groupby(["node_id", "output"], sort=False):
                fig = F.paired_errors_plot(tidy, col, a, b, node_id, output)
                _save(fig, f"paired_errors_{_slug(col)}_{_slug(b)}_vs_{_slug(a)}__{_slug(output)}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _kv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in [p for p in re.split(r"[;,]", text or "") if p.strip()]:
        key, _, value = part.partition("=")
        out[key.strip()] = value.strip()
    return out


def _csv(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def _prediction_sources(items: Sequence[str], source_col: str) -> List[Any]:
    """``label=path`` items become sources tagged with ``source_col = label``."""
    out: List[Any] = []
    for item in items:
        label, sep, path = item.partition("=")
        if sep and not Path(item).exists() and Path(path).exists():
            out.append(load_predictions(path, meta={source_col: label}))
        else:
            out.append(item)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="with_annotated_dataset",
        description="Evaluate COMPASS predictions against an annotated dataset.",
    )
    sub = parser.add_subparsers(dest="command")
    ev = sub.add_parser("evaluate", help="tidy table, metrics with bootstrap CIs, and figures",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    ev.add_argument("--predictions", nargs="+", required=True,
                    help="result folder(s), .jsonl/.json records, .csv/.tsv, or a tidy CSV; 'label=path' tags a source")
    ev.add_argument("--annotations", default=None, help="annotation JSON/JSONL/CSV/TSV (omit when truth is embedded)")
    ev.add_argument("--task-spec", default=None, help="PredictionTaskSpec JSON (inferred when absent)")
    ev.add_argument("--group-by", default="", help="comma-separated grouping columns, for example tier,predictor")
    ev.add_argument("--out", required=True, help="output directory")
    ev.add_argument("--id-column", default=None, help="participant id column of CSV inputs")
    ev.add_argument("--annotation-columns", default="", help="target=column pairs, for example total=TOTAL,root=dx")
    ev.add_argument("--targets", default="", help="comma-separated annotation target columns")
    ev.add_argument("--prediction-columns", default="", help="target=column pairs for wide prediction tables")
    ev.add_argument("--path-groups", default="", help="names of folder levels under a results root, for example tier,predictor")
    ev.add_argument("--source-col", default="source", help="column name for 'label=path' prediction sources")
    ev.add_argument("--compare", action="append", default=None,
                    help="paired comparison 'column:a,b' (repeatable); 'column' alone compares the first value with the others")
    ev.add_argument("--repeat-col", default="auto", help="repeat column for test-retest ('auto', a column name, or 'none')")
    ev.add_argument("--multilabel", default="", help="group=node1|node2 multi-label groups of binary nodes, ';' separated")
    ev.add_argument("--positive-label", default="", help="node=label pairs setting the positive class of binary nodes")
    ev.add_argument("--expected-participants", choices=["observed", "annotations"], default="observed",
                    help="'annotations' counts annotated participants without a prediction as missing in every group")
    ev.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ev.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ev.add_argument("--ci", type=float, default=DEFAULT_CI)
    ev.add_argument("--no-figures", action="store_true")
    ev.add_argument("--figure-formats", default="png", help="comma-separated, for example png,pdf")
    ev.add_argument("--tier-col", default="tier")
    ev.add_argument("--tier-order", default="", help="comma-separated tier order for by-tier figures")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "evaluate":
        parser.print_help()
        return 2
    multilabel = {}
    for part in [p for p in args.multilabel.split(";") if p.strip()]:
        name, _, nodes = part.partition("=")
        multilabel[name.strip()] = [n.strip() for n in nodes.split("|") if n.strip()]
    sources = _prediction_sources(args.predictions, args.source_col)
    repeat_col = None if str(args.repeat_col).lower() == "none" else args.repeat_col
    group_by = _csv(args.group_by)
    try:
        result = evaluate(
            sources if len(sources) > 1 else sources[0],
            args.annotations,
            task_spec=args.task_spec,
            group_by=group_by,
            out_dir=args.out,
            compare=args.compare,
            repeat_col=repeat_col,
            multilabel_groups=multilabel or None,
            positive_labels=_kv(args.positive_label) or None,
            expected_participants=args.expected_participants,
            id_column=args.id_column,
            annotation_columns=_kv(args.annotation_columns) or None,
            annotation_targets=_csv(args.targets) or None,
            prediction_columns=_kv(args.prediction_columns) or None,
            path_groups=_csv(args.path_groups),
            n_boot=args.n_boot,
            seed=args.seed,
            ci=args.ci,
            figures=not args.no_figures,
            figure_formats=_csv(args.figure_formats) or ["png"],
            tier_col=args.tier_col,
            tier_order=_csv(args.tier_order) or None,
        )
    except (ValueError, FileNotFoundError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    s = result.summary
    print(f"Evaluated {s['n_rows']} rows, {s['n_participants']} participants; status {s['status_counts']}")
    headline = result.metrics[result.metrics["metric"].isin(["pearson_r", "mae", "balanced_accuracy", "auroc", "macro_f1"])]
    if not headline.empty:
        cols = [*s["group_by"], "node_id", "output", "metric", "value", "ci_low", "ci_high", "n", "n_missing_prediction"]
        with pd.option_context("display.width", 160, "display.max_rows", 60, "display.float_format", "{:.3f}".format):
            print(headline[[c for c in cols if c in headline.columns]].to_string(index=False))
    print(f"Wrote {len(result.files)} tables and {len(result.figures)} figures to {args.out}")
    return 0
