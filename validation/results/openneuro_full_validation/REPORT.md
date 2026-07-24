# OpenNeuro full-validation report

Generated from the completed seeded validation design on 2026-07-24.

## Completion and validity

All 453 planned jobs completed with a structurally valid prediction:

| Dataset | Subjects | Tasks | Valid jobs |
|---|---:|---:|---:|
| INTELLIGENCE | 100 | 1 | 100/100 |
| PSYCHOSIS | 143 | 1 | 143/143 |
| NUMERACY | 105 | 2 | 210/210 |

The design used seed `20260723`, temperature `0.0`, one assigned tier per
subject, and the five-provider panel recorded in `summary.json`. The full run
had no dollar cap. OpenRouter usage increased from $151.080000 to $158.332657,
for a recorded full-run cost of $7.252657.

Final validation rejected any missing, non-numeric, non-finite, off-scale, or
invalid-label output. It also checked exact dataset and provider allocations
and rejected conflicting duplicate job keys. The final accepted set contains
no such violations.

## Predictive results

### Intelligence

Performance shows modest positive association with ground truth across all
four outputs:

| Output | n | MAE | RMSE | Bias | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| Crystallised | 100 | 8.777 | 10.754 | 0.396 | 0.381 | 0.392 |
| Fluid | 100 | 22.389 | 27.413 | -3.956 | 0.342 | 0.318 |
| Total | 100 | 34.812 | 42.824 | -4.201 | 0.420 | 0.378 |
| Memory | 100 | 7.479 | 9.588 | -0.553 | 0.323 | 0.299 |

The MAEs are not directly comparable across outputs because their native
scales differ. The total score has the strongest correlation but also the
largest native-scale error.

### Numeracy

Both numeracy tasks show modest signal with errors around 0.73 population
standard deviations:

| Output | Evaluable n | MAE | RMSE | Bias | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| Approximate numeracy | 104 | 0.734 | 0.965 | -0.095 | 0.344 | 0.222 |
| Precise numeracy | 104 | 0.729 | 1.042 | -0.404 | 0.335 | 0.403 |

Precise-numeracy predictions are systematically lower than ground truth. One
job per task lacks an evaluable ground-truth target, so metric denominators are
104 even though all 105 predictions per task are structurally valid.

### Psychosis

Diagnosis performance is:

| n | Accuracy | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---:|---:|---:|---:|---:|---:|
| 143 | 0.517 | 0.559 | 0.247 | 0.871 | 0.612 |

The large sensitivity-specificity gap shows a strong tendency to predict the
control class. Clinical-profile tier results were descriptively strongest
(balanced accuracy 0.643, AUROC 0.798), while the demographic-only tier had
only 0.050 sensitivity. These are between-subject cells and should not be read
as a causal ablation result.

Symptom-score metrics use the 79 subjects with available targets. Errors and
biases were:

| Output | MAE | Bias | Pearson r |
|---|---:|---:|---:|
| BPRS total | 17.319 | -9.009 | -0.092 |
| SAPS hallucinations | 1.706 | -0.760 | -0.259 |
| SAPS delusions | 1.912 | -1.403 | -0.197 |
| SAPS bizarre behavior | 1.107 | -0.392 | -0.002 |
| SAPS thought disorder | 1.093 | -0.298 | -0.051 |
| SANS affective flattening | 1.089 | -0.618 | 0.130 |
| SANS alogia | 0.981 | -0.310 | 0.061 |
| SANS avolition | 1.679 | -0.974 | -0.006 |
| SANS anhedonia | 1.669 | -0.935 | -0.002 |
| SANS attention | 1.059 | -0.475 | 0.246 |

The broadly negative biases and weak correlations indicate symptom severity
is generally underestimated and poorly ranked. These results are not adequate
for clinical decision-making.

## Tier and provider observations

Normalized-MAE tables make outputs with different native scales comparable
before macro averaging. Descriptively:

- Intelligence macro normalized MAE was lowest for demographics (0.724) and
  psychological data (0.726), and highest for multimodal-full data (0.954).
- Numeracy macro normalized MAE was lowest for demographics (0.610) and
  highest for fine lesion data (1.268).
- Psychosis symptom macro normalized MAE was lowest for demographic and
  socioeconomic data (1.099), whereas diagnosis discrimination was strongest
  in the clinical-profile tier.
- Provider differences were material, but provider cells contain different
  randomly assigned subjects. They are useful monitoring summaries, not
  direct head-to-head model rankings.

Because this is a seeded between-subject design rather than a within-subject
ablation, none of these tier or provider differences alone establishes that
adding a modality causes performance to improve or worsen.

## Recovery and execution audit

The accepted predictions were never clipped or filled with fabricated values.
Invalid or incomplete responses were quarantined and rerun with the originally
assigned provider:

- 39 successful rows retain one or more prior attempt errors.
- Two additional failed-cache rows are preserved in `summary.json` and matched
  to a valid prediction recovered in an isolated shard.
- A GPT NUMERACY prediction of `-22.0` for `sub-048` was rejected because it
  was outside the declared `[-5, 5]` population-Z range. The job was rerun and
  its valid replacement, `0.01`, is the only version in the accepted set.
- Predictor schema validation was moved inside the LLM retry loop so a
  syntactically valid response that omitted a required multivariate output was
  not accepted.

Poolside had the shortest median execution times in every dataset and no
recorded recovered jobs. Nex required the most recovery in PSYCHOSIS
(10 jobs with attached prior failures; median 714.9 seconds). These execution
figures describe this run and provider state, not intrinsic model reliability.

## Limitations

- This is one seeded validation run. Provider behavior and hosted model
  versions can change.
- Temperature `0.0` reduces sampling variability but does not make remote
  inference, tool execution, or provider routing perfectly deterministic.
- Tier and provider comparisons are between subjects. A within-subject design
  is required for a clean modality ablation or paired provider comparison.
- Ground-truth availability limits some metrics: 104/105 per NUMERACY task and
  79/143 for PSYCHOSIS symptom scores.
- Recovery was based only on execution and output validity, never on whether a
  prediction agreed with hidden truth. Predictive weaknesses therefore remain
  visible in the reported metrics.
- These outputs are research validation results, not clinical predictions for
  care decisions.

## Artifacts

- `predictions.json`: all 453 accepted job records.
- `predictions_tidy.csv`: long-form diagnosis and regression predictions.
- `*_predictions.csv`: dataset-specific long-form predictions.
- `metrics/`: overall, tier, provider, and normalized-error tables.
- `figures/`: performance summaries for each dataset.
- `execution_by_dataset_model.csv`: timing, attempt, and recovery summary.
- `summary.json`: design, spend, counts, metrics, invalid-discard record, and
  recovery audit.
