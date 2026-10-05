# Annotation Templates

These templates define recommended annotation payload structures for validation.

## Files

- `examples/binary_targets_example.json`
  - Ground-truth labels for `--prediction_type binary`
- `examples/multiclass_annotations_example.json`
  - Ground-truth class labels for `--prediction_type multiclass`
- `examples/regression_univariate_annotations_example.json`
  - Exactly one numeric output per participant
- `examples/regression_multivariate_annotations_example.json`
  - Two or more numeric outputs per participant
- `examples/hierarchical_annotations_example.json`
  - Node-wise mixed labels/values for hierarchical validation
- `examples/multilabel_annotations_example.json`
  - Label sets per participant (multi-label groups; an empty list means no label)
- `examples/annotations_table_example.csv`
  - The same information as a table: one row per participant, one column per
    target, other columns (here `site`) become grouping covariates
- `examples/predictions_tidy_example.csv`
  - Predictions as a long table (one row per participant x output; an empty
    value is a missing prediction)

The legacy CLIs (`run_validation_metrics.py`, `detailed_analysis.py`) read the
JSON templates. The evaluation library (`run_evaluation.py`, `evaluate()`)
reads all of them; see the parent README.

## Notes

- Participant IDs can be stored as `eid`, `participant_id`, or object keys.
- Optional grouping keys (`disorder`, `group`, `cohort`, `phenotype_group`) are used for per-group analysis.
- For non-binary modes, pass the JSON via `--annotations_json`.
- Hierarchical mode requires consistent node schema across participants (same node IDs, node modes, and regression output keys per node).

## Recommended JSON Envelope

Use either a dictionary keyed by participant ID or an `annotations` list.

Dictionary form:

```json
{
  "SUBJ_001": {"label": "CASE", "disorder": "GROUP_1"},
  "SUBJ_002": {"label": "CONTROL", "disorder": "GROUP_1"}
}
```

List form:

```json
{
  "annotations": [
    {"eid": "SUBJ_001", "label": "CASE", "disorder": "GROUP_1"},
    {"eid": "SUBJ_002", "regression": {"total_iq": 101.4}, "disorder": "GROUP_1"}
  ]
}
```

For non-binary modes, replace `label` with the appropriate payload:
- multiclass: `label`
- regression: `regression` (or `values`)
- hierarchical: `nodes`
- multi-label: `multilabel: {group: [labels]}`

## Tables (evaluation library)

CSV or TSV with a participant id column (`participant_id`, `eid`, `subject`,
or `--id-column`). Map targets to columns with
`--annotation-columns target=column`, for example
`--annotation-columns root=diagnosis,total_score=total_score`; numeric
targets are regression outputs, text targets are class labels.
