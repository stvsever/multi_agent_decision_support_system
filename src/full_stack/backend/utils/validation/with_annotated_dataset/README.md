# Annotated dataset evaluation

Evaluate COMPASS predictions against an annotated dataset, whatever the input
and output structure: binary, multiclass, multi-label, hierarchical and mixed
task trees, univariate and multivariate regression. Everything is normalized
into one tidy long table, scored per output or node and per group with seeded
bootstrap 95% confidence intervals, and drawn as a standard set of figures.

Dependencies: numpy, pandas, scipy, matplotlib. The engine package is not
imported, so the library also runs on result files copied to another machine.

## Quick start

Python:

```python
from src.full_stack.backend.utils.validation.with_annotated_dataset import evaluate

result = evaluate(
    "results/participant_runs",          # COMPASS result folders, JSON/JSONL records, CSV/TSV, DataFrame ...
    "annotations.json",                  # JSON template, CSV/TSV, dict; None when truth is in the records
    group_by=["tier", "predictor"],      # any columns of the records or annotations
    out_dir="results/analysis/evaluation",
)
result.metrics        # long table: group x node x output x metric, value, ci_low, ci_high, counts
result.metrics_wide   # one row per group x node x output
result.tidy           # the normalized long table
```

Command line (same options, from the repository root):

```bash
python -m src.full_stack.backend.utils.validation.with_annotated_dataset evaluate \
    --predictions results/participant_runs \
    --annotations annotations.json \
    --group-by tier,predictor \
    --out results/analysis/evaluation
```

`python run_evaluation.py evaluate ...` inside this folder does the same
without importing the rest of the backend. `python -m ... evaluate --help`
lists every option.

## Inputs

**Predictions** (`--predictions`, one or more; `label=path` tags a source,
the column name is set with `--source-col`):

| Source | What is read |
|---|---|
| Folder of COMPASS result folders | every `report_<id>.json` below it (nested `root_prediction` tree, task spec), plus `performance_report_<id>.json` for `predictor_kind`, `predictor_model`, `route`, `total_tokens`, `duration_seconds`; a `participant_*` folder without a report is a failed run. `--path-groups tier,predictor` names the folder levels of `root/<tier>/<predictor>/participant_<id>/`. |
| `.jsonl` / `.json` records | one record per prediction job: an id (`participant_id`, `eid`, `subject`, ...), a prediction block (`root_prediction` tree, `flat_predictions`, `classifications: {node: {label, probs}}`, `regression: {output: value}`, `label` + `probs`, `multilabel: {group: [labels] or {label: p}}`) and any scalar fields as grouping columns (`tier`, `predictor`, `provider`, `repeat`, `route`, ...). `ok: false` or a failed `status` marks a failed run. Embedded truth (`ground_truth`, `labels_truth`, `diagnosis`, `truth`) belongs to its own record (ids may repeat across datasets) and fills whatever an annotation file does not cover; the annotation file wins. |
| `.csv` / `.tsv` | long layout (`participant_id`, `output`, optional `node_id`, `pred_value` or `pred_label`, optional `probs` JSON or `prob_<label>` or `p_positive` columns, optional `true_value`/`true_label`, grouping columns) or wide layout (one column per output, `--prediction-columns name=column`). A tidy table written by this library is read back as is. |
| Python | a DataFrame, a dict or list of dicts, pydantic results, or `load_predictions(...)` sets (concatenated, `meta={"provider": "x"}` adds a constant column). |

**Annotations** (`--annotations`): the JSON templates in
`annotation_templates/examples/` (dict keyed by participant id or
`{"annotations": [...]}` with `label`, `regression`/`values`, `nodes`,
`multilabel`), JSONL, or a CSV/TSV with `--id-column` and
`--annotation-columns target=column` (for example `total=TOTAL_SCORE,root=dx`).
`--id-column` applies to both tables; when they name the id differently, load
them separately in Python (`load_predictions(..., id_column=...)`,
`load_annotations(..., id_column=...)`) and pass the results to `evaluate`.
Columns that are not targets (for example `site`, `sex`, `disorder`) become
covariates usable in `--group-by`. Participant ids are matched exactly first,
then leniently (`participant_ID01`, `sub-01` and `01` match).

**Task structure** (`--task-spec`): a COMPASS `PredictionTaskSpec` (object,
dict or JSON file). Without it, the spec inside COMPASS reports is used, else
the structure is inferred (numeric targets are regression, label targets are
classification, list-valued targets are multi-label). For a binary node the
positive class is `--positive-label node=label`, else the label that does not
look negative (`control`, `healthy`, `absent`, ...), else `class_labels[0]`.
`CASE`/`CONTROL` truth labels map onto the node's two class labels.

## The tidy table

One row per prediction instance x node x output:

| Column | Meaning |
|---|---|
| grouping columns | everything carried from records and annotations (`tier`, `predictor`, ...) |
| `instance_id`, `participant_id` | one engine run for one participant |
| `node_id`, `parent_node_id`, `depth` | position in the task tree |
| `output` | regression output, the node id for a classification node, or the label of a multi-label group |
| `kind` | `regression`, `binary`, `multiclass`, `multilabel` |
| `status` | `ok`, `missing_prediction`, `missing_truth`, `missing_both` |
| `true_value`, `pred_value`, `error` | regression (and 0/1 for multi-label); `error = pred - true` |
| `true_label`, `pred_label`, `correct` | classification |
| `positive_label`, `p_positive`, `p_true`, `probs` | probabilities (renormalized over the node's labels) |

Missing predictions are rows, not gaps: every metric row reports `n` (scored),
`n_total`, `n_missing_prediction`, `n_missing_truth` and `coverage`, and
classification adds `accuracy_missing_as_wrong`. With
`--expected-participants annotations`, every annotated participant without a
prediction in a group is added as a missing prediction there.

## Metrics

All intervals are percentile intervals of a seeded bootstrap that resamples
participants (default 2000 replicates, `--n-boot`, `--seed`); repeated rows of
one participant stay together. Predictions are scored as given, never
recalibrated.

- **Regression** (per output, plus `__macro__` means for multi-output nodes):
  `pearson_r`, `spearman_rho`, `r2` (`1 - SS_res/SS_tot`, can be negative),
  `mae`, `rmse`, `nmae` (MAE / SD of truth), `bias` (mean pred - true),
  `ols_slope` and `ols_intercept` (predicted regressed on true), `lin_ccc`.
- **Classification**: `accuracy`, `balanced_accuracy`, `macro_f1`, `kappa`,
  `accuracy_missing_as_wrong`; binary adds `sensitivity`, `specificity`, `ppv`,
  `npv`, `f1`, `mcc`; with probabilities `auroc` (binary, or macro one-vs-rest),
  `brier` (binary, or summed over classes) and, for binary nodes, `ece`
  (expected calibration error, 10 equal-width bins). Per-class precision, recall and
  F1 with CIs in `per_class`, confusion counts (with a `(missing)` column) in
  `confusion`.
- **Multi-label** (list-valued targets, or `--multilabel group=node1|node2` over
  binary nodes): per label `f1`, `sensitivity`, `specificity`,
  `balanced_accuracy`, `prevalence`, `auroc`, `brier`; overall `macro_f1`,
  `micro_f1`, `micro_precision`, `micro_recall`, `macro_balanced_accuracy`,
  `macro_auroc`, `hamming_loss`, `subset_accuracy`.
- **Hierarchy** (task trees with more than one node): `node_coverage`,
  `exact_match` over all classification nodes, `path_accuracy` per root-to-leaf
  path, and `child_accuracy_given_parent_correct` / `_wrong` per edge.
- **Paired comparison** (`--compare predictor:llm,decision`, repeatable): on
  participants scored under both configurations, `value_a`, `value_b`,
  `diff = b - a` with a paired bootstrap CI, `p_boot`, and `better`.
- **Test-retest** (column `repeat`, or `--repeat-col`): agreement of the
  predictions across repeats: `icc_2_1`, `icc_3_1`, `within_sd` for regression,
  `cohen_kappa` (2 repeats) or `fleiss_kappa` and `percent_agreement` for labels.

## Outputs

`tidy.csv`, `metrics_long.csv/.json`, `metrics_wide.csv/.json`,
`per_class.csv`, `confusion.csv`, `multilabel.csv/.json`, `hierarchy.csv/.json`,
`comparison.csv/.json`, `test_retest.csv/.json`, `summary.json`, and
`figures/` (PNG, `--figure-formats png,pdf` for vector copies):

- `scatter_<node>__<output>`: density scatter of predicted vs true per group,
  identity line, OLS fit with 95% confidence band, stats box with CIs
- `residuals_<node>__<output>`: residuals vs truth and Bland-Altman agreement
- `forest_<metric>`: metric with 95% CI for every group and output
- `by_tier_<output>__<metric>`: metric by tier, one line per configuration
  (when `tier` is a grouping column; `--tier-col`, `--tier-order`)
- `confusion_<node>__<group>`, `roc_<node>`, `calibration_<node>` (binary
  reliability diagram), `multilabel_f1`
- `paired_<col>_<b>_vs_<a>__<metric>`, `paired_errors_...`

One fixed-order colorblind-safe palette is used everywhere (a configuration
keeps its color across figures), with a single-hue blue ramp for magnitude.

## Using the pieces

```python
from src.full_stack.backend.utils.validation.with_annotated_dataset import (
    build_tidy, compute_metrics, compare_configurations, test_retest, make_figures,
)
from src.full_stack.backend.utils.validation.with_annotated_dataset.core import figures

tidy, structure = build_tidy(records, annotations, group_by=["tier"], return_structure=True)
tables = compute_metrics(tidy, ["tier", "predictor"], structure=structure, n_boot=2000, seed=1)
paired = compare_configurations(tidy, "predictor", "llm", "decision", group_by=["tier"])
retest = test_retest(tidy, "repeat", group_by=["tier"])
fig = figures.scatter_grid(tidy, "root", "total_score", facet_cols=["tier"], metrics=tables.metrics)
figures.save_figure(fig, "scatter_total_score", formats=("png", "pdf"))
```

`compute_metrics` needs only the columns `participant_id`, `node_id`,
`output`, `kind`, `status` and the value or label columns of each kind, so any
table built elsewhere in that shape can be scored and drawn.

## Legacy entry points

`run_validation_metrics.py` and `detailed_analysis.py` keep their mode-specific
interface (`--prediction_type binary|multiclass|regression_univariate|
regression_multivariate|hierarchical`, `--targets_file` or
`--annotations_json`); see `annotation_templates/README.md`.
`utils/batch_run.py --run_validation` runs both legacy scripts and this
library (`--validation_engine legacy|evaluate|both`, `--validation_group_by`).
